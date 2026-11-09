"""End-to-end run: TLC trip records + MTA data -> monthly aggregates -> difference-in-differences -> site and README.

python -m congestion.pipeline              # real data (first run downloads every month since 2023; later runs only new ones)
python -m congestion.pipeline --fixture    # offline run on synthetic data (tests and CI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from . import analysis as A
from . import mta, tlc, zones
from .config import POLICY_MONTH, Config
from .source import get_source, months_between

log = logging.getLogger("congestion")


def load_zones(cfg: Config, src) -> tuple[pd.DataFrame, dict]:
    """Zone lookup and polygons, cached in data/ (they rarely change)."""
    cache = cfg.data_dir / "zones"
    cache.mkdir(parents=True, exist_ok=True)
    lk, gj = cache / "taxi_zone_lookup.csv", cache / "taxi_zones.geojson"
    if not lk.exists():
        src.zone_lookup().to_csv(lk, index=False)
    if not gj.exists():
        zones.save_geojson(zones.shapes_to_geojson(src.zone_shapes()), gj)
    return zones.clean_lookup(pd.read_csv(lk)), json.loads(gj.read_text())


def analyze(cfg: Config, src) -> dict:
    t0 = time.time()
    latest = src.latest_month()
    months = months_between(cfg.start_month, latest)
    processed = tlc.update(cfg, src, months)
    lookup, geo = load_zones(cfg, src)
    m = mta.load_all(src, cfg)

    con = A.connect(cfg)
    official = zones.official_cbd(m.get("cbd_rows", []))
    derived, fee_by_zone = zones.fee_cbd(con, min_trips=5 if cfg.fixture else 50)
    cbd, cbd_info = zones.choose_cbd(official, derived)
    zt = zones.regions(lookup, cbd)
    A.set_zones(con, zt)
    log.info("Zone definition: %s (%d zones); agreement official/fees %s", cbd_info["used"], len(cbd),
             cbd_info["agreement_official_fees"])  # fmt: skip

    full = A.complete_months(con, cfg.services)
    rel = A.relation_month(con)
    rel = rel[pd.to_datetime(rel["month"]).isin(full)].assign(month=lambda d: pd.to_datetime(d["month"]))
    speeds = {p: A.region_speed_month(con, p) for p in ("weekday_peak", "weekend_peak", "overnight")}
    for p, s in speeds.items():
        speeds[p] = s.assign(month=pd.to_datetime(s["month"]))[lambda d: d["month"].isin(full)]
    panels = {"all": A.zone_panel(con, cfg), "weekend_peak": A.zone_panel(con, cfg, "weekend_peak"),
              "overnight": A.zone_panel(con, cfg, "overnight"), "yellow": A.zone_panel(con, cfg, service="yellow"),
              "fhvhv": A.zone_panel(con, cfg, service="fhvhv")}  # fmt: skip
    panels = {k: v[v["month"].isin(full)] for k, v in panels.items()}
    results = A.speed_results(panels, full)
    es, pretrend_p = A.event_study(panels["all"])
    es_trips, pretrend_trips = A.event_study(panels["all"], outcome="pickups")
    trips_did = A.did(panels["all"], outcome="pickups", label="Pickups: inside vs. Manhattan above 60th St")
    main = results[0]
    log.info("Speed effect %.1f%% [%.1f%%, %.1f%%]; pre-trend %s", main["effect"] * 100, main["lo"] * 100,
             main["hi"] * 100, pretrend_p)  # fmt: skip

    crz = A.crz_summary(m["crz"]) if "crz" in m else None
    bus = A.bus_did(m["bus"]) if "bus" in m else None
    zone_change = A.zone_changes(con, panels["all"], full)
    within = rel[rel["period"] == "weekday_peak"]
    wk = speeds["weekday_peak"]
    raw = {r: A.yoy(wk[wk["region"] == r], "mph", full, weight="trips") for r in (A.INSIDE, A.UPPER, A.OUTER)}
    rel_yoy = {r: A.yoy(rel[rel["relation"] == r].groupby("month", as_index=False)["trips"].sum(), "trips", full)
               for r in A.RELATIONS}  # fmt: skip
    taxi_official = m.get("taxi_speeds")
    zone_month = con.execute("""
        SELECT date_trunc('month', date)::DATE AS month, service, zone, sum(pickups) AS pickups,
               sum(dropoffs) AS dropoffs
        FROM zoneday GROUP BY ALL ORDER BY 1, 2, 3
    """).df()  # fmt: skip
    zone_month = zone_month[pd.to_datetime(zone_month["month"]).isin(full)]
    est = [{"specification": r["label"], "effect": r["effect"], "lo": r["lo"], "hi": r["hi"], "p_value": r["p"],
            "zones_inside": r["zones_inside"], "zones_control": r["zones_control"], "observations": r["obs"]}
           for r in [*results, trips_did] if r.get("ok")]  # fmt: skip
    if bus:
        est.append({"specification": "Bus speeds: segments inside vs. outside the zone", "effect": bus["effect"],
                    "lo": bus["lo"], "hi": bus["hi"], "p_value": bus["p"], "zones_inside": None,
                    "zones_control": None, "observations": bus["units"]})  # fmt: skip
    load_log = tlc.load_log(cfg)
    con.close()
    return {
        "months": full, "latest_month": latest, "processed_now": processed, "load_log": load_log,
        "zones": zt, "geo": geo, "cbd_info": cbd_info, "fee_by_zone": fee_by_zone, "relations": rel,
        "within_peak": within, "speeds": speeds, "panels": panels, "results": results, "event_study": es,
        "pretrend_p": pretrend_p, "event_study_trips": es_trips, "pretrend_trips_p": pretrend_trips,
        "trips_did": trips_did, "crz": crz, "bus": bus, "zone_change": zone_change, "raw_speed_yoy": raw,
        "relation_yoy": rel_yoy, "fee_check": A.fee_check(rel), "taxi_official": taxi_official,
        "mta_notes": m["notes"], "seconds": round(time.time() - t0), "policy_month": POLICY_MONTH,
        "zone_month": zone_month, "estimates": pd.DataFrame(est), "bus_raw": m.get("bus"),
        "daily_detail": m["crz"]["daily"] if "crz" in m else None,
    }  # fmt: skip


def run(cfg: Config) -> dict:
    from . import explore
    from .readme import update_readme
    from .report import build_site

    res = analyze(cfg, get_source(cfg))
    summary = build_site(cfg, res)
    tables = explore.tables(res)
    meta = explore.export(tables, cfg.site_dir / "data")
    runtime = explore.copy_runtime(cfg.vendor_dir, cfg.site_dir)
    explore.write_page(cfg.site_dir, meta, summary["data_through"], runtime, cfg.repo_url, cfg.fixture)
    summary["browser_sql"] = {"tables": len(meta), "runtime": runtime}
    (cfg.site_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    if not cfg.fixture:
        update_readme(cfg.readme_path, summary)
    log.info("Done in %ss. Site: %s", res["seconds"], cfg.site_dir / "index.html")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", action="store_true", help="offline run on synthetic data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.fixture:
        base = Path(os.environ.get("CPI_FIXTURE_DIR", "/tmp/cpi-fixture")).resolve()
        shutil.rmtree(base, ignore_errors=True)
        os.environ.update({"CPI_FIXTURE": "1", "CPI_DATA_DIR": str(base / "data"), "CPI_SITE_DIR": str(base / "site")})
    s = run(Config())
    print(
        json.dumps(
            {
                "data_through": s["data_through"],
                "speed_effect": s["speed_effect"]["effect"],
                "zones_inside": s["zones_inside"],
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
