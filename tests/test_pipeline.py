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
    main = s["speed_effect"]
    assert main["lo"] < fixture.PLANTED_SPEED < main["hi"]
    placebo = next(r for r in s["robustness"] if r["label"].startswith("Placebo"))
    assert placebo["lo"] < 0 < placebo["hi"]
    assert s["pickups_effect"]["lo"] < fixture.PLANTED_PICKUPS < s["pickups_effect"]["hi"]
    assert s["bus_effect"]["lo"] < fixture.PLANTED_BUS < s["bus_effect"]["hi"]
    assert s["pretrend"]["p"] > 0.05
    assert s["entries"]["change"] == pytest.approx(-0.05, abs=0.02)  # the fixture's 2026 entries are 5% lower
    assert s["zones_inside"] == len(fixture.CBD) and s["zone_agreement"] == 1.0


def test_incremental_rerun_loads_nothing(run):
    cfg, s = run
    again = pipeline.analyze(cfg, fixture.FixtureSource(cfg))
    assert again["processed_now"] == [] and again["results"][0]["effect"] == pytest.approx(s["speed_effect"]["effect"])


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
    assert "Speed of trips inside the zone" in text and text.count("<!-- RESULTS:START -->") == 1
