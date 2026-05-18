"""Ingest arXiv daily mailing list emails via IMAP and emit JSONL.

Required environment variables:
  EMAIL_USERNAME, EMAIL_APP_PASSWORD, EMAIL_IMAP_HOST, EMAIL_IMAP_PORT,
  ARXIV_EMAIL_FROM, ARXIV_EMAIL_SUBJECT_KEYWORD

Optional:
  INCLUDE_REPLACEMENTS    default false
  INCLUDE_CROSS_LISTINGS  default true
"""
from __future__ import annotations

import argparse
import email
import imaplib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from email.header import decode_header, make_header
from typing import Dict, Iterable, List, Optional, Tuple


ARXIV_ID_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
# Block-start: arXiv: must be at the very beginning of the line (no leading
# spaces).  This prevents false splits on in-text citations like
# "[Smith, arXiv:2603.15792]" or indented abstract references.
BLOCK_START_RE = re.compile(r"^arXiv:\d{4}\.\d{4,5}", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^-{20,}\s*$")
ABS_URL_RE = re.compile(r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE)

SOURCE_PRIORITY = {"new": 0, "cross-listing": 1, "replacement": 2}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _decode_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _get_body(msg: email.message.Message) -> str:
    plain_parts: List[str] = []
    html_parts: List[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp.lower():
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload is not None:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "\n".join(html_parts)
        soup = BeautifulSoup("\n".join(html_parts), "html.parser")
        return soup.get_text("\n")
    return ""


def _split_blocks(body: str) -> List[str]:
    """Split mail body into per-paper blocks keyed by arXiv: id lines.

    Only lines where 'arXiv:' is at column 0 are treated as block starts.
    In-text citations like '[Smith, arXiv:2603.15792]' or indented abstract
    references are intentionally ignored.
    """
    lines = body.splitlines()
    starts: List[int] = []
    for i, line in enumerate(lines):
        if BLOCK_START_RE.match(line):
            starts.append(i)
    blocks: List[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        # Trim trailing separator lines
        block_lines = lines[start:end]
        while block_lines and SEPARATOR_RE.match(block_lines[-1].strip()):
            block_lines.pop()
        blocks.append("\n".join(block_lines))
    return blocks


def _detect_source_type(block: str) -> str:
    low = block.lower()
    if "replaced with revised version" in low:
        return "replacement"
    if "cross-list" in low or "(*cross-listing*)" in low or "cross-listing" in low:
        return "cross-listing"
    return "new"


FIELD_LABELS = ["Title", "Authors", "Categories", "Comments", "Report-no",
                "Journal-ref", "MSC-class", "ACM-class", "Subj-class",
                "DOI", "License", "Date"]
FIELD_LINE_RE = re.compile(r"^\s*(" + "|".join(FIELD_LABELS) + r")\s*:\s*(.*)$")


def _parse_block(block: str) -> Optional[Dict]:
    m = ARXIV_ID_RE.search(block)
    if not m:
        return None
    arxiv_id = m.group(1)

    lines = block.splitlines()

    fields: Dict[str, str] = {}
    current: Optional[str] = None
    abstract_lines: List[str] = []
    abstract_mode = False
    # The abstract is delimited by "\\" lines. After the Title/Authors/...
    # block there is usually a "\\" line, then abstract text, then another "\\".
    backslash_count = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        # arXiv mailing list uses lines like "\\" (or "\\ ( https://...abs/.. )")
        # to delimit the abstract section. The first such line ends the
        # field block and starts the abstract; the second ends the abstract.
        is_backslash_marker = stripped.startswith("\\\\") or stripped == "\\"
        if is_backslash_marker:
            backslash_count += 1
            if backslash_count == 1:
                abstract_mode = True
                current = None
                continue
            else:
                abstract_mode = False
                current = None
                continue

        if abstract_mode:
            abstract_lines.append(line)
            continue

        fm = FIELD_LINE_RE.match(line)
        if fm:
            current = fm.group(1)
            fields[current] = fm.group(2).strip()
            continue
        # Continuation of a multi-line field value
        if current and stripped:
            fields[current] = (fields[current] + " " + stripped).strip()

    # Sometimes the abstract is preceded by a single "\\" and the trailing
    # link line includes the closing "\\". If we never entered abstract_mode,
    # try a fallback: take everything between the last field line and the
    # link line.
    abstract_text = "\n".join(abstract_lines).strip()
    if not abstract_text:
        # Fallback: find region after fields & before abs URL
        joined = "\n".join(lines)
        url_match = re.search(r"\(\s*https?://arxiv\.org/abs/\d{4}\.\d{4,5}", joined)
        if url_match:
            head = joined[: url_match.start()]
            # The abstract should be the last paragraph after a blank line
            parts = re.split(r"\n\s*\n", head)
            if len(parts) >= 2:
                abstract_text = parts[-1].strip()

    title = fields.get("Title", "").strip()
    authors_raw = fields.get("Authors", "").strip()
    categories_raw = fields.get("Categories", "").strip()
    comment = fields.get("Comments", "").strip()

    # Authors: split by ", " and " and "
    authors: List[str] = []
    if authors_raw:
        # Normalize " and " => ", "
        normalized = re.sub(r"\s+and\s+", ", ", authors_raw)
        for part in normalized.split(","):
            name = part.strip()
            if name:
                authors.append(name)

    categories: List[str] = [c for c in categories_raw.split() if c]

    summary = re.sub(r"\s+", " ", abstract_text).strip()

    if not title or not summary:
        print(
            f"[WARN] Skipping arXiv:{arxiv_id} -- missing "
            f"{'title' if not title else ''}{' and ' if not title and not summary else ''}"
            f"{'summary' if not summary else ''}",
            file=sys.stderr,
        )
        return None

    return {
        "id": arxiv_id,
        "title": re.sub(r"\s+", " ", title).strip(),
        "authors": authors,
        "categories": categories,
        "summary": summary,
        "comment": re.sub(r"\s+", " ", comment).strip(),
        "abs": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        "source_type": _detect_source_type(block),
    }


def _completeness(item: Dict) -> int:
    score = 0
    for f in ("title", "summary"):
        if item.get(f):
            score += 2
    for f in ("authors", "categories"):
        if item.get(f):
            score += 1
    if item.get("comment"):
        score += 1
    return score


def _merge_items(items: Iterable[Dict]) -> List[Dict]:
    by_id: Dict[str, Dict] = {}
    for it in items:
        prev = by_id.get(it["id"])
        if prev is None:
            by_id[it["id"]] = it
            continue
        # Prefer better source_type
        cur_p = SOURCE_PRIORITY.get(it.get("source_type", "new"), 99)
        prev_p = SOURCE_PRIORITY.get(prev.get("source_type", "new"), 99)
        if cur_p < prev_p:
            by_id[it["id"]] = it
        elif cur_p == prev_p:
            if _completeness(it) > _completeness(prev):
                by_id[it["id"]] = it
    return list(by_id.values())


def fetch_emails(target_date: datetime) -> List[email.message.Message]:
    user = os.environ.get("EMAIL_USERNAME")
    pwd = os.environ.get("EMAIL_APP_PASSWORD")
    host = os.environ.get("EMAIL_IMAP_HOST")
    port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
    sender = os.environ.get("ARXIV_EMAIL_FROM", "")
    subject_kw = os.environ.get("ARXIV_EMAIL_SUBJECT_KEYWORD", "")

    if not (user and pwd and host):
        print("[ERROR] Missing IMAP credentials (EMAIL_USERNAME / EMAIL_APP_PASSWORD / EMAIL_IMAP_HOST).", file=sys.stderr)
        sys.exit(1)

    # arXiv mailing list is in US Eastern; allow +/- 1 day window.
    since = (target_date - timedelta(days=1)).strftime("%d-%b-%Y")
    before = (target_date + timedelta(days=2)).strftime("%d-%b-%Y")

    print(f"[INFO] Connecting IMAP {host}:{port} as {user}", file=sys.stderr)
    M = imaplib.IMAP4_SSL(host, port)
    try:
        M.login(user, pwd)
        M.select("INBOX", readonly=True)

        criteria: List[str] = ["SINCE", since, "BEFORE", before]
        if sender:
            criteria += ["FROM", f'"{sender}"']
        if subject_kw:
            criteria += ["SUBJECT", f'"{subject_kw}"']

        typ, data = M.search(None, *criteria)
        if typ != "OK" or not data or not data[0]:
            print(f"[WARN] IMAP search returned no results. criteria={criteria}", file=sys.stderr)
            return []

        ids = data[0].split()
        print(f"[INFO] Found {len(ids)} candidate emails", file=sys.stderr)
        msgs: List[email.message.Message] = []
        for mid in ids:
            typ, msg_data = M.fetch(mid, "(RFC822)")
            if typ != "OK" or not msg_data:
                continue
            for part in msg_data:
                if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], (bytes, bytearray)):
                    msgs.append(email.message_from_bytes(part[1]))
                    break
        return msgs
    finally:
        try:
            M.close()
        except Exception:
            pass
        M.logout()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="Target date YYYY-MM-DD (UTC)")
    p.add_argument("--out", required=True, help="Output JSONL path")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"[ERROR] Invalid --date: {args.date}", file=sys.stderr)
        sys.exit(1)

    include_replacements = _env_bool("INCLUDE_REPLACEMENTS", False)
    include_cross = _env_bool("INCLUDE_CROSS_LISTINGS", True)

    msgs = fetch_emails(target_date)
    if not msgs:
        print("[ERROR] No arXiv mailing list emails found in the search window.", file=sys.stderr)
        sys.exit(1)

    all_items: List[Dict] = []
    for msg in msgs:
        subject = _decode_header(msg.get("Subject"))
        body = _get_body(msg)
        if not body:
            print(f"[WARN] Empty body for subject: {subject}", file=sys.stderr)
            continue
        blocks = _split_blocks(body)
        for block in blocks:
            parsed = _parse_block(block)
            if parsed is not None:
                all_items.append(parsed)

    merged = _merge_items(all_items)

    filtered: List[Dict] = []
    for it in merged:
        st = it.get("source_type", "new")
        if st == "replacement" and not include_replacements:
            continue
        if st == "cross-listing" and not include_cross:
            continue
        filtered.append(it)

    if not filtered:
        print("[ERROR] Parsed 0 papers from emails. Check ARXIV_EMAIL_FROM / SUBJECT_KEYWORD or mailbox state.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in filtered:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(filtered)} items to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
