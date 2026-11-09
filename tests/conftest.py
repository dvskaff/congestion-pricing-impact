import os
from dataclasses import replace

import pytest

os.environ.setdefault("CPI_FIXTURE", "1")

from congestion.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return replace(Config(), fixture=True, min_zone_month_trips=20, data_dir=tmp_path / "data",
                   site_dir=tmp_path / "site", readme_path=tmp_path / "README.md",
                   vendor_dir=tmp_path / "no-vendor")  # fmt: skip
