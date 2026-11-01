"""Step 2 - Scrape documents (fetch only, no parsing).

Reads sources.csv, downloads each page politely and saves:
  data/01_raw/<slug>.html        raw page
  data/01_raw/<slug>.meta.json   url, slug, source_type, scheme_name, fetched_at, status, sha256
  data/01_raw/fetch_report.json  one line per source: fetched / unchanged / skipped / failed
"""
import csv
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from src.config import get_config, path

IST = timezone(timedelta(hours=5, minutes=30))
REQUIRED_COLUMNS = {"url", "slug", "source_type", "scheme_name"}


def read_sources() -> list[dict]:
    with open(path("sources.csv"), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"sources.csv is missing columns: {sorted(missing)}")
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]
    rows = [r for r in rows if r["url"] and not r["url"].startswith("#")]
    slugs = [r["slug"] for r in rows]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    if dupes:
        raise ValueError(f"Duplicate slugs in sources.csv: {sorted(dupes)}")
    return rows


class RobotsChecker:
    """Caches one robots.txt per domain. If robots.txt can't be read, allow."""

    def __init__(self, client: httpx.Client, user_agent: str):
        self.client, self.ua, self.cache = client, user_agent, {}

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.cache:
            rp = RobotFileParser()
            try:
                r = self.client.get(base + "/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self.cache[base] = rp
        return self.cache[base].can_fetch(self.ua, url)


def fetch(client: httpx.Client, url: str, retries: int) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            r = client.get(url)
            r.raise_for_status()
            return r
        except httpx.HTTPError:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def run(force: bool = False) -> list[dict]:
    cfg = get_config()["scraping"]
    raw_dir = path("data/01_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = []

    headers = {"User-Agent": cfg["user_agent"], "Accept-Language": "en-IN,en;q=0.9"}
    with httpx.Client(headers=headers, timeout=cfg["timeout_s"], follow_redirects=True) as client:
        robots = RobotsChecker(client, cfg["user_agent"])
        for src in read_sources():
            slug, url = src["slug"], src["url"]
            entry = {"slug": slug, "url": url}

            if cfg.get("respect_robots", True) and not robots.allowed(url):
                report.append({**entry, "result": "skipped", "reason": "disallowed by robots.txt"})
                print(f"  SKIP     {slug} (robots.txt)")
                continue

            try:
                r = fetch(client, url, cfg["retries"])
            except httpx.HTTPError as e:
                report.append({**entry, "result": "failed", "reason": str(e)})
                print(f"  FAILED   {slug}: {e}")
                continue

            sha = hashlib.sha256(r.content).hexdigest()
            meta_path = raw_dir / f"{slug}.meta.json"
            if not force and meta_path.exists():
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                if old.get("content_sha256") == sha:
                    report.append({**entry, "result": "unchanged"})
                    print(f"  SAME     {slug}")
                    time.sleep(cfg["delay_s"])
                    continue

            (raw_dir / f"{slug}.html").write_text(r.text, encoding="utf-8")
            meta = {
                **src,
                "final_url": str(r.url),
                "fetched_at": datetime.now(IST).isoformat(timespec="seconds"),
                "status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "bytes": len(r.content),
                "content_sha256": sha,
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            report.append({**entry, "result": "fetched", "bytes": len(r.content)})
            print(f"  FETCHED  {slug} ({len(r.content) // 1024} KB)")
            time.sleep(cfg["delay_s"])

    (raw_dir / "fetch_report.json").write_text(json.dumps(
        {"run_at": datetime.now(IST).isoformat(timespec="seconds"), "sources": report},
        indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {k: sum(1 for e in report if e["result"] == k)
              for k in ("fetched", "unchanged", "skipped", "failed")}
    print(f"  Done: {counts}")
    return report


if __name__ == "__main__":
    run()
