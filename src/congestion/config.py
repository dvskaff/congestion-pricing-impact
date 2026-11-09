"""Project settings. Paths and a few knobs can be overridden with environment variables (tests and CI)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ[env]).resolve() if os.environ.get(env) else default


# New York's Congestion Relief Zone (CRZ): Manhattan south of and including 60th Street. Tolling started on
# Sunday, January 5, 2025. Peak hours: weekdays 5 a.m. to 9 p.m., weekends 9 a.m. to 9 p.m. Taxis and for-hire
# vehicles pay a per-trip charge at all hours for trips that start, end or are completed in the zone.
POLICY_START = date(2025, 1, 5)
POLICY_MONTH = "2025-01"

# Fallback list of TLC taxi zones in the zone (Manhattan south of 60th Street), used only if neither the MTA's own
# list (data.ny.gov yfdc-w5jh) nor the per-trip charges in the trip records can be read. The pipeline reports how
# the three definitions agree.
FALLBACK_CBD_ZONES = frozenset({
    4, 12, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 144, 148, 158, 161, 162, 163, 164,
    170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249, 261,
})  # fmt: skip


@dataclass
class Config:
    # ---- data window: two full years before the toll (for seasonality and a placebo test) and everything since
    start_month: str = field(default_factory=lambda: os.environ.get("CPI_START_MONTH", "2023-01"))
    services: tuple[str, ...] = ("yellow", "green", "fhvhv")
    tlc_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/{service}_tripdata_{month}.parquet"
    zones_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    lookup_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    socrata_url: str = "https://data.ny.gov/resource/{dataset}.json"
    # MTA datasets on data.ny.gov
    crz_entries: str = "t6yz-b64h"  # Congestion Relief Zone vehicle entries, 10-minute blocks, from 2025-01-05
    bus_speeds: str = "r6db-kkzj"  # MTA bus speeds inside / outside the zone, from 2023
    taxi_speeds: str = "6p29-6xqn"  # MTA's own taxi + FHV speed estimates for the zone, from Oct 2019
    cbd_zones: str = "yfdc-w5jh"  # TLC taxi zones the MTA counts as inside the zone
    user_agent: str = "congestion-pricing-impact (github.com/dvskaff/congestion-pricing-impact)"

    # ---- analysis
    min_zone_month_trips: int = 150  # zone-months with fewer speed-valid internal trips are left out of the panel
    min_zone_coverage: float = 0.9  # zones must have data in this share of months (a nearly balanced panel)
    fixture: bool = field(default_factory=lambda: os.environ.get("CPI_FIXTURE") == "1")

    # ---- paths
    data_dir: Path = field(default_factory=lambda: _path("CPI_DATA_DIR", ROOT / "data"))
    site_dir: Path = field(default_factory=lambda: _path("CPI_SITE_DIR", ROOT / "site"))
    readme_path: Path = field(default_factory=lambda: _path("CPI_README", ROOT / "README.md"))
    vendor_dir: Path = field(default_factory=lambda: _path("CPI_VENDOR_DIR", ROOT / "web" / "vendor"))
    repo_url: str = "https://github.com/dvskaff/congestion-pricing-impact"
    site_url: str = "https://dvskaff.github.io/congestion-pricing-impact/"

    def __post_init__(self):
        if self.fixture:
            self.min_zone_month_trips = 20

    @property
    def agg_dir(self) -> Path:
        return self.data_dir / "agg"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"
