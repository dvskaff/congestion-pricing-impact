"""End to end on synthetic data with planted effects: the analysis must find them, and only them."""

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from congestion import explore, fixture, pipeline
from congestion.readme import update_readme

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from dataclasses import replace

    from congestion.config import Config

    tmp = tmp_path_factory.mktemp("run")
    cfg = replace(Config(), fixture=True, min_zone_month_trips=20, data_dir=tmp / "data", site_dir=tmp / "site",
                  readme_path=tmp / "README.md", vendor_dir=tmp / "no-vendor")  # fmt: skip
    return cfg, pipeline.run(cfg)


def test_planted_effects_are_recovered(run):
    cfg, s = run
    sp = s["speed"]
    for key in ("first", "trend", "pooled"):  # no pre-trend in the fixture: every version finds the planted +8%
        assert sp[key]["lo"] < fixture.PLANTED_SPEED < sp[key]["hi"], key
    assert sp["first"]["year"] == 2025 and sp["second"] is None  # the fixture ends in mid-2025
    assert sp["placebo"]["year"] == 2023 and sp["placebo"]["lo"] < 0 < sp["placebo"]["hi"]
    assert s["pickups"]["first"]["lo"] < fixture.PLANTED_PICKUPS < s["pickups"]["first"]["hi"]
    assert s["bus"]["first"]["lo"] < fixture.PLANTED_BUS < s["bus"]["first"]["hi"]
    assert all(r["label"].endswith("2025 vs. 2024") for r in s["robustness"])
    assert s["pretrend"]["p"] > 0.05
    assert s["entries"]["change"] == pytest.approx(-0.05, abs=0.02)  # the fixture's 2026 entries are 5% lower
    assert s["zones_inside"] == len(fixture.CBD) and s["zone_agreement"] == 1.0


def test_incremental_rerun_loads_nothing(run):
    cfg, s = run
    again = pipeline.analyze(cfg, fixture.FixtureSource(cfg))
    assert again["processed_now"] == []
    assert again["speed"]["first"]["effect"] == pytest.approx(s["speed"]["first"]["effect"])


def test_site_tables_and_readme(run):
    cfg, s = run
    site = cfg.site_dir
    page = (site / "index.html").read_text()
    assert "SYNTHETIC TEST DATA" in page and 'id="zonemap"' in page and 'href="sql.html"' in page
    assert json.loads((site / "results.json").read_text())["synthetic"] is True
    meta = json.loads((site / "data" / "tables.json").read_text())["tables"]
    names = {t["name"] for t in meta}
    assert {"zones", "zone_month", "zone_speeds", "trips_by_relation", "estimates", "vehicle_entries"} <= names
    con = duckdb.connect()
    for t in meta:
        con.execute(explore.create_sql(t, str(site / t["file"])))
        assert con.execute(f"SELECT count(*) FROM {t['name']}").fetchall()[0][0] == t["rows"] > 0
    for _, sql in explore.EXAMPLES:
        con.execute(sql).fetchall()
    assert "castDecimalToDouble" in (site / "sql.html").read_text()

    shutil.copy(ROOT / "README.md", cfg.readme_path)
    update_readme(cfg.readme_path, s)
    text = cfg.readme_path.read_text()
    assert "Speed of taxi and app-ride trips inside the zone" in text and "First toll year: 2025 vs. 2024" in text
    assert text.count("<!-- RESULTS:START -->") == 1


def test_trend_adjustment_handles_a_pre_existing_drift():
    """If the inside zones were already slowing 3% a year before the toll, a plain before/after comparison hides a real
    effect; the trend-adjusted estimate recovers it and the placebo year flags the problem."""
    import numpy as np
    import pandas as pd

    from congestion import analysis as A

    rng = np.random.default_rng(0)
    months = pd.date_range("2023-01-01", "2026-06-01", freq="MS")
    rows = []
    for z in range(40):
        inside = z < 20
        for m in months:
            t = (m - pd.Timestamp("2025-01-01")).days / 365.25
            log_speed = 2.2 + 0.1 * np.sin(m.month) + (-0.03 * t if inside else 0)
            log_speed += (0.06 if inside and m >= pd.Timestamp("2025-01-01") else 0) + rng.normal(0, 0.01)
            rows.append({"zone": z, "month": m, "mph": np.exp(log_speed), "speed_trips": 500, "pickups": 1000,
                         "region": A.INSIDE if inside else A.UPPER})  # fmt: skip
    res = A.effects(A.frame(pd.DataFrame(rows)))
    assert res["trend"]["lo"] < np.exp(0.06) - 1 < res["trend"]["hi"]
    assert res["placebo"]["effect"] > 0.02  # 2023 sits above 2024: the drift shows up before any toll
    assert res["first"]["effect"] < res["trend"]["effect"]
