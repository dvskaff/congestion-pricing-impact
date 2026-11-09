"""Trip records -> small monthly aggregates (DuckDB). Each raw file (up to ~20 million rides for Uber/Lyft) is read
once, condensed and deleted; only the aggregates are kept, so later runs only process newly published months.

Per month and service three tables are written:
* od:      pickup zone x dropoff zone x time period -> trips, trips charged the congestion fee, fee total, and the
           miles and hours of trips with a plausible speed (for average speeds)
* zoneday: date x zone -> pickups and dropoffs (for weekly counts)
* hourly:  pickup zone x hour x weekday/weekend -> trips and fee-charged trips (when the per-trip fee applies)

Time periods follow the toll's own hours: weekday peak (5 a.m.-9 p.m.), weekend peak (9 a.m.-9 p.m.), overnight.
TLC timestamps are local New York time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)
KINDS = ("od", "zoneday", "hourly")
PICKUP = ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime")
DROPOFF = ("tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime")
DISTANCE = ("trip_distance", "trip_miles")
PERIOD_SQL = """CASE
    WHEN dayofweek(pu_ts) IN (0, 6) THEN CASE WHEN hour(pu_ts) BETWEEN 9 AND 20 THEN 'weekend_peak' ELSE 'overnight' END
    ELSE CASE WHEN hour(pu_ts) BETWEEN 5 AND 20 THEN 'weekday_peak' ELSE 'overnight' END END"""
# A trip counts toward average speeds if it lasted 1 minute to 3 hours, covered 0.1-100 miles and averaged 0.5-75 mph.
SPEED_OK = "hours BETWEEN 1.0 / 60 AND 3 AND miles BETWEEN 0.1 AND 100 AND miles / hours BETWEEN 0.5 AND 75"


def _pick(cols: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((cols[n] for n in names if n in cols), None)


def aggregate(path: Path, service: str, month: str, out_dir: Path) -> dict:
    """Write the three aggregates for one file; returns row counts (read, kept) for the data-quality log."""
    con = duckdb.connect()
    con.execute("SET threads TO 4; SET memory_limit = '5GB'; SET preserve_insertion_order = false")
    src = f"read_parquet('{path.as_posix()}')"
    cols = {c[0].lower(): c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
    pu, do = _pick(cols, PICKUP), _pick(cols, DROPOFF)
    dist = _pick(cols, DISTANCE)
    if not (pu and do and dist and "pulocationid" in cols and "dolocationid" in cols):
        raise ValueError(f"{path.name}: unexpected columns {sorted(cols)}")
    hours = (
        f'TRY_CAST("{cols["trip_time"]}" AS DOUBLE) / 3600'
        if "trip_time" in cols
        else f'epoch("{do}"::TIMESTAMP - "{pu}"::TIMESTAMP) / 3600'
    )
    fee = f'COALESCE(TRY_CAST("{cols["cbd_congestion_fee"]}" AS DOUBLE), 0)' if "cbd_congestion_fee" in cols else "0.0"
    start = pd.Period(month, "M").start_time
    end = (pd.Period(month, "M") + 1).start_time
    con.execute(f"""
        CREATE TEMP TABLE t AS
        SELECT "{pu}"::TIMESTAMP AS pu_ts,
               TRY_CAST("{cols["pulocationid"]}" AS INTEGER) AS pu,
               TRY_CAST("{cols["dolocationid"]}" AS INTEGER) AS dz,
               TRY_CAST("{dist}" AS DOUBLE) AS miles,
               {hours} AS hours,
               {fee} AS fee
        FROM {src}
        WHERE "{pu}"::TIMESTAMP >= TIMESTAMP '{start}' AND "{pu}"::TIMESTAMP < TIMESTAMP '{end}'
    """)  # fmt: skip
    read = con.execute(f"SELECT count(*) FROM {src}").fetchall()[0][0]
    con.execute("DELETE FROM t WHERE pu NOT BETWEEN 1 AND 263 OR dz NOT BETWEEN 1 AND 263 OR pu IS NULL OR dz IS NULL")
    kept = con.execute("SELECT count(*) FROM t").fetchall()[0][0]
    m = f"DATE '{start.date()}'"
    tables = {
        "od": f"""
            SELECT {m} AS month, '{service}' AS service, pu, dz AS "do", {PERIOD_SQL} AS period,
                   count(*) AS trips, count(*) FILTER (WHERE fee > 0) AS fee_trips, sum(fee) AS fee_sum,
                   count(*) FILTER (WHERE {SPEED_OK}) AS speed_trips,
                   COALESCE(sum(miles) FILTER (WHERE {SPEED_OK}), 0) AS miles,
                   COALESCE(sum(hours) FILTER (WHERE {SPEED_OK}), 0) AS hours
            FROM t GROUP BY ALL""",
        "zoneday": f"""
            SELECT day AS date, '{service}' AS service, zone, sum(p)::BIGINT AS pickups, sum(d)::BIGINT AS dropoffs
            FROM (SELECT pu_ts::DATE AS day, pu AS zone, 1 AS p, 0 AS d FROM t
                  UNION ALL SELECT pu_ts::DATE, dz, 0, 1 FROM t) GROUP BY ALL""",
        "hourly": f"""
            SELECT {m} AS month, '{service}' AS service, pu AS zone, hour(pu_ts) AS hour,
                   dayofweek(pu_ts) IN (0, 6) AS weekend, count(*) AS trips,
                   count(*) FILTER (WHERE fee > 0) AS fee_trips
            FROM t GROUP BY ALL""",
    }  # fmt: skip
    for kind, sql in tables.items():
        dest = out_dir / kind / f"{service}_{month}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")  # write, then rename: an interrupted run never leaves a half-written file
        con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT parquet)")
        tmp.replace(dest)
    con.close()
    return {"service": service, "month": month, "rows_read": int(read), "rows_kept": int(kept),
            "has_fee_column": "cbd_congestion_fee" in cols}  # fmt: skip


def done(out_dir: Path, service: str, month: str) -> bool:
    return all((out_dir / k / f"{service}_{month}.parquet").exists() for k in KINDS)


def update(cfg, src, months: list[str]) -> list[dict]:
    """Download and aggregate every (service, month) not already aggregated. Returns the log of processed files."""
    log_path = cfg.agg_dir / "load_log.json"
    history = json.loads(log_path.read_text()) if log_path.exists() else []
    processed = []
    for month in months:
        for service in cfg.services:
            if done(cfg.agg_dir, service, month):
                continue
            raw = cfg.raw_dir / f"{service}_{month}.parquet"
            path = src.download_tlc(service, month, raw)
            if path is None:
                log.warning("TLC %s %s not published; skipped", service, month)
                continue
            size = path.stat().st_size
            info = aggregate(path, service, month, cfg.agg_dir)
            path.unlink()
            info["bytes"] = size
            processed.append(info)
            log.info("Aggregated %s %s: %s of %s rows kept", service, month, f"{info['rows_kept']:,}",
                     f"{info['rows_read']:,}")  # fmt: skip
    if processed:
        cfg.agg_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(history + processed, indent=1))
    return processed


def load(cfg, kind: str) -> pd.DataFrame:
    files = sorted((cfg.agg_dir / kind).glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No {kind} aggregates in {cfg.agg_dir}")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet({[f.as_posix() for f in files]})").df()
    con.close()
    return df


def load_log(cfg) -> pd.DataFrame:
    p = cfg.agg_dir / "load_log.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else pd.DataFrame()
