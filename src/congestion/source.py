"""Where the data comes from: NYC TLC trip records (monthly Parquet files on the TLC's public CDN), the TLC taxi zones,
and MTA datasets on data.ny.gov (Socrata API). No account or key is needed for any of them.

`get_source` returns the real source or, for tests, the synthetic one in fixture.py; both have the same methods.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)


def months_between(start: str, end: str) -> list[str]:
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


class HttpSource:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.s = requests.Session()
        self.s.headers["User-Agent"] = cfg.user_agent

    def _get(self, url: str, *, stream: bool = False, params: dict | None = None, tries: int = 4):
        for i in range(tries):
            try:
                r = self.s.get(url, params=params, timeout=(20, 300), stream=stream)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                if i == tries - 1:
                    raise
                wait = 5 * 2**i
                log.warning("%s: %s; retrying in %ss", url, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ TLC trip records
    def tlc_exists(self, service: str, month: str) -> bool:
        url = self.cfg.tlc_url.format(service=service, month=month)
        try:
            r = self.s.head(url, timeout=30, allow_redirects=True)
            return r.status_code == 200 and int(r.headers.get("Content-Length", "1")) > 0
        except requests.RequestException:
            return False

    def latest_month(self, service: str = "yellow", today: date | None = None) -> str:
        """Newest month the TLC has published (it posts each month with a delay of about two months)."""
        p = pd.Period(today or date.today(), "M")
        for _ in range(9):
            p -= 1
            if self.tlc_exists(service, str(p)):
                return str(p)
        raise RuntimeError("No TLC trip file found for the last 9 months")

    def download_tlc(self, service: str, month: str, dest: Path) -> Path | None:
        url = self.cfg.tlc_url.format(service=service, month=month)
        r = self._get(url, stream=True)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
        return dest

    # ------------------------------------------------------------------ zones
    def zone_lookup(self) -> pd.DataFrame:
        r = self._get(self.cfg.lookup_url)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))

    def zone_shapes(self) -> bytes:
        r = self._get(self.cfg.zones_url)
        r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------ data.ny.gov
    def socrata(self, dataset: str, params: dict | None = None, page: int = 50_000, max_rows: int = 2_000_000):
        """All rows of a SoQL query, paged. Raises on an invalid query (the message names the bad field)."""
        url = self.cfg.socrata_url.format(dataset=dataset)
        rows, offset = [], 0
        while offset < max_rows:
            q = {**(params or {}), "$limit": page, "$offset": offset}
            r = self._get(url, params=q)
            if r.status_code >= 400:
                raise RuntimeError(f"data.ny.gov {dataset}: HTTP {r.status_code}: {r.text[:300]}")
            batch = r.json()
            rows += batch
            if len(batch) < page:
                break
            offset += page
        return rows


def get_source(cfg: Config):
    if cfg.fixture:
        from .fixture import FixtureSource

        return FixtureSource(cfg)
    return HttpSource(cfg)
