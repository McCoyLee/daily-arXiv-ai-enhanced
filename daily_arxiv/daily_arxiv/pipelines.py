import time

import arxiv


class DailyArxivPipeline:
    def __init__(self):
        self.page_size = 1
        self.client = arxiv.Client(page_size=1, delay_seconds=6.0, num_retries=5)
        self.max_backoff_retries = 5

    def _need_fetch(self, item: dict) -> bool:
        required = ["title", "summary", "authors", "categories"]
        return any(not item.get(field) for field in required)

    def _fetch_with_retry(self, arxiv_id: str):
        last_exc = None
        for attempt in range(1, self.max_backoff_retries + 1):
            try:
                search = arxiv.Search(id_list=[arxiv_id])
                return next(self.client.results(search))
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "429" in msg or "rate limit" in msg or "httperror" in msg:
                    sleep_sec = min(30 * (2 ** (attempt - 1)), 120)
                    print(
                        f"[WARN] arXiv fetch rate-limited for {arxiv_id}, attempt {attempt}/{self.max_backoff_retries}, "
                        f"sleeping {sleep_sec}s. err={exc}"
                    )
                    time.sleep(sleep_sec)
                    continue
                print(f"[WARN] arXiv fetch failed for {arxiv_id}, attempt {attempt}/{self.max_backoff_retries}. err={exc}")
                time.sleep(min(5 * attempt, 20))
        raise last_exc

    def process_item(self, item: dict, spider):
        arxiv_id = item["id"]
        item["pdf"] = f"https://arxiv.org/pdf/{arxiv_id}"
        item["abs"] = f"https://arxiv.org/abs/{arxiv_id}"

        if not self._need_fetch(item):
            return item

        try:
            paper = self._fetch_with_retry(arxiv_id)
            item["authors"] = item.get("authors") or [a.name for a in paper.authors]
            item["title"] = item.get("title") or paper.title
            item["categories"] = item.get("categories") or paper.categories
            item["comment"] = item.get("comment") or paper.comment
            item["summary"] = item.get("summary") or paper.summary
        except Exception as exc:
            spider.logger.warning(
                "Failed to fetch metadata for arXiv id %s after retries: %s. Keeping existing fields.",
                arxiv_id,
                exc,
            )
        return item
