class DailyArxivPipeline:
    """Pass-through pipeline.

    Metadata (title/authors/categories/summary) is now expected to be already
    populated by the spider or upstream ingest (e.g. email_ingest.py). We only
    fill in the canonical abs/pdf URLs derived from the arXiv id.
    """

    def process_item(self, item: dict, spider):
        arxiv_id = item["id"]
        item["pdf"] = f"https://arxiv.org/pdf/{arxiv_id}"
        item["abs"] = f"https://arxiv.org/abs/{arxiv_id}"
        return item
