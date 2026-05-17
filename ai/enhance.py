import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import dotenv
import requests
from langchain_openai import ChatOpenAI
from tqdm import tqdm

if os.path.exists('.env'):
    dotenv.load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


def is_sensitive(content: str) -> bool:
    """Fail-open sensitive check: checker outage should not drop all papers."""
    spam_check_url = os.environ.get("SPAM_CHECK_URL", "")
    if not spam_check_url:
        return False

    try:
        resp = requests.post(
            spam_check_url,
            json={"text": content},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("sensitive", False)
        print(f"[WARN] Sensitive check failed with status {resp.status_code}, url={spam_check_url}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[WARN] Sensitive check error (fail-open), url={spam_check_url}, error={exc}", file=sys.stderr)
        return False


def check_github_code(content: str) -> Dict:
    code_info: Dict = {}
    github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
    match = re.search(github_pattern, content)
    if match:
        owner, repo = match.groups()
        repo = repo.rstrip(".git").rstrip(".,)")
        code_info["code_url"] = f"https://github.com/{owner}/{repo}"

        github_token = os.environ.get("TOKEN_GITHUB")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
        except Exception:
            pass
        return code_info

    github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
    match_io = re.search(github_io_pattern, content)
    if match_io:
        code_info["code_url"] = match_io.group(0).rstrip(".,)")
    return code_info


def parse_ai_json(raw_content: str, item_id: str, defaults: Dict[str, str]) -> Tuple[Dict[str, str], Optional[str]]:
    if not raw_content:
        return defaults.copy(), f"Empty content for {item_id}"

    try:
        data = json.loads(raw_content)
    except Exception as exc:
        return defaults.copy(), f"Invalid JSON for {item_id}: {exc}; raw={raw_content[:500]}"

    ai_obj = {}
    missing_fields = []
    for field, default_val in defaults.items():
        val = data.get(field)
        if not val or not isinstance(val, str):
            ai_obj[field] = default_val
            missing_fields.append(field)
        else:
            ai_obj[field] = val.strip()

    if missing_fields:
        return ai_obj, f"Missing/invalid fields for {item_id}: {missing_fields}"
    return ai_obj, None


def process_single_item(llm, item: Dict, language: str, defaults: Dict[str, str], system_prompt: str, human_prompt_template: str) -> Tuple[Optional[Dict], bool]:
    if is_sensitive(item.get("summary", "")):
        return None, False

    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    try:
        human_prompt = human_prompt_template.format(language=language, content=item.get("summary", ""))
        response = llm.invoke([("system", system_prompt), ("human", human_prompt)])
        raw = response.content if hasattr(response, "content") else ""
        if isinstance(raw, list):
            raw = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in raw)

        ai_data, warn = parse_ai_json(raw, item.get("id", "unknown"), defaults)
        if warn:
            print(f"[WARN] {warn}", file=sys.stderr)
        item["AI"] = ai_data
    except Exception as exc:
        print(f"[ERROR] AI generation failed for {item.get('id', 'unknown')}: {exc}", file=sys.stderr)
        item["AI"] = defaults.copy()

    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None, False

    failed = item["AI"].get("tldr") == defaults["tldr"]
    return item, failed


def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> Tuple[List[Optional[Dict]], int]:
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=2048,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    print(f"Connect to: {model_name}", file=sys.stderr)

    system_prompt = "You are an expert research assistant. Return valid json only. Do not output markdown or code fences."
    human_prompt_template = (
        "Summarize the following arXiv abstract in {language}. Output MUST be json with keys: "
        "tldr, motivation, method, result, conclusion. Include all keys even if uncertain. "
        "Use concise text.\n"
        "json example:\n"
        "{\n"
        "  \"tldr\": \"...\",\n"
        "  \"motivation\": \"...\",\n"
        "  \"method\": \"...\",\n"
        "  \"result\": \"...\",\n"
        "  \"conclusion\": \"...\"\n"
        "}\n"
        "Abstract:\n{content}"
    )
    defaults = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed",
    }

    processed_data: List[Optional[Dict]] = [None] * len(data)
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_item, llm, item, language, defaults, system_prompt, human_prompt_template): idx
            for idx, item in enumerate(data)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Processing items"):
            idx = future_to_idx[future]
            try:
                result, failed = future.result()
                processed_data[idx] = result
                if failed:
                    fail_count += 1
            except Exception as exc:
                print(f"[ERROR] Item at index {idx} exception: {exc}", file=sys.stderr)
                item = data[idx]
                item["AI"] = defaults.copy()
                processed_data[idx] = item
                fail_count += 1

    return processed_data, fail_count


def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")

    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f"Removed existing file: {target_file}", file=sys.stderr)

    data: List[Dict] = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_data.append(item)

    processed_data, fail_count = process_all_items(unique_data, model_name, language, args.max_workers)

    total_items = len(unique_data)
    with open(target_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    fail_ratio = (fail_count / total_items) if total_items else 0
    print(f"AI enhancement completed. failed={fail_count}, total={total_items}, ratio={fail_ratio:.2%}", file=sys.stderr)
    if total_items > 0 and fail_ratio > 0.5:
        print("[ERROR] AI enhancement failure ratio exceeds 50%, exiting with code 1.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
