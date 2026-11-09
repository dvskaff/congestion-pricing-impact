"""MTA datasets on data.ny.gov, read with server-side SoQL aggregation so only small tables are downloaded.

Field names are discovered from one sample row and matched by meaning rather than assumed, and every dataset is
optional: if one can't be read, the analysis carries on without it and the report says so.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find(fields: list[str], *patterns: str, exclude: tuple[str, ...] = ()) -> str | None:
    """First field whose normalized name contains every token of a pattern (patterns tried in order)."""
    for pat in patterns:
        toks = [_norm(t) for t in pat.split()]
        for f in fields:
            n = _norm(f)
            if all(t in n for t in toks) and not any(_norm(e) in n for e in exclude):
                return f
    return None


def fields(src, dataset: str) -> list[str]:
    rows = src.socrata(dataset, {"$limit": 1}, page=1, max_rows=1)
    return list(rows[0]) if rows else []


def _day(v) -> pd.Series:
    return pd.to_datetime(v, errors="coerce").dt.normalize()


# --------------------------------------------------------------------------- vehicle entries into the zone
def crz_entries(src, dataset: str) -> dict:
    """Daily entries by vehicle class, entry point and toll period, plus entries by hour of the day."""
    f = fields(src, dataset)
    date = find(f, "toll date", "date", exclude=("week",))
    hour = find(f, "toll hour")
    cls, group = find(f, "vehicle class"), find(f, "detection group", "entry point", "detection region")
    period, n = find(f, "time period", "period"), find(f, "crz entries", "entries", exclude=("excluded",))
    excl = find(f, "excluded")
    if not (date and cls and n):
        raise RuntimeError(f"CRZ entries: unrecognized fields {f}")
    dims = [d for d in (date, cls, group, period) if d]
    sel = ", ".join(dims) + f", sum({n}) AS entries" + (f", sum({excl}) AS excluded" if excl else "")
    daily = pd.DataFrame(src.socrata(dataset, {"$select": sel, "$group": ", ".join(dims), "$order": ", ".join(dims)}))
    daily = daily.rename(columns={date: "date", cls: "vehicle_class", group or "_g": "entry_point",
                                  period or "_p": "period"})  # fmt: skip
    daily["date"] = _day(daily["date"])
    for c in ("entries", "excluded"):
        if c in daily:
            daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0)
    out = {"daily": daily, "fields": f}
    if hour:
        h = pd.DataFrame(src.socrata(dataset, {"$select": f"{hour}, sum({n}) AS entries", "$group": hour,
                                               "$order": hour}))  # fmt: skip
        h["ts"] = pd.to_datetime(h[hour], errors="coerce")
        h["entries"] = pd.to_numeric(h["entries"], errors="coerce").fillna(0)
        out["hourly"] = h[["ts", "entries"]].dropna()
    return out


# --------------------------------------------------------------------------- bus speeds inside / outside the zone
def bus_speeds(src, dataset: str) -> pd.DataFrame:
    """Monthly average bus speed by route and CBD relation (weighted by trips if the dataset has a trip count)."""
    f = fields(src, dataset)
    month, rel = find(f, "month", "date"), find(f, "cbd relation", "relation", "cbd")
    route, speed = find(f, "route id", "route"), find(f, "average road speed", "speed")
    count = find(f, "total trips", "trip count", "trips", "count", exclude=("speed",))
    if not (month and rel and speed):
        raise RuntimeError(f"Bus speeds: unrecognized fields {f}")
    dims = [d for d in (month, route, rel) if d]
    if count:
        sel = ", ".join(dims) + f", sum({speed} * {count}) AS sw, sum({count}) AS n"
    else:
        sel = ", ".join(dims) + f", avg({speed}) AS sw, count(*) AS n"
    try:
        rows = src.socrata(dataset, {"$select": sel, "$group": ", ".join(dims), "$order": ", ".join(dims)})
    except RuntimeError:  # weighted sum not accepted: plain averages
        sel = ", ".join(dims) + f", avg({speed}) AS sw, count(*) AS n"
        rows = src.socrata(dataset, {"$select": sel, "$group": ", ".join(dims), "$order": ", ".join(dims)})
        count = None
    df = pd.DataFrame(rows).rename(columns={month: "month", rel: "relation", route or "_r": "route"})
    df["sw"], df["n"] = pd.to_numeric(df["sw"], errors="coerce"), pd.to_numeric(df["n"], errors="coerce")
    df["speed"] = df["sw"] / df["n"] if count else df["sw"]
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "route" not in df:
        df["route"] = "all"
    return df.dropna(subset=["month", "speed"])[["month", "route", "relation", "speed", "n"]]


def relation_role(value: str) -> str | None:
    """'treated' for bus segments inside the zone, 'control' for those outside it, None for anything else."""
    v = str(value).lower()
    if any(k in v for k in ("partial", "cross", "through", "both", "border")):
        return None  # segments that are partly inside: neither treated nor control
    if any(k in v for k in ("outside", "non-cbd", "non cbd", "noncbd", "not in")):
        return "control"
    if any(k in v for k in ("inside", "within", "in cbd", "internal")) or v.strip() in ("cbd", "in"):
        return "treated"
    return None


# --------------------------------------------------------------------------- the MTA's own taxi speed estimates
def taxi_speeds(src, dataset: str) -> pd.DataFrame:
    f = fields(src, dataset)
    month, zone = find(f, "month", "date"), find(f, "zone", "area", "region")
    speed = find(f, "zonal speed", "speed")
    if not (month and zone and speed):
        raise RuntimeError(f"Taxi speeds: unrecognized fields {f}")
    rows = src.socrata(dataset, {"$select": f"{month}, {zone}, avg({speed}) AS speed", "$group": f"{month}, {zone}",
                                 "$order": f"{month}, {zone}"})  # fmt: skip
    df = pd.DataFrame(rows).rename(columns={month: "month", zone: "area"})
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return df.dropna(subset=["month", "speed"])


def load_all(src, cfg) -> dict:
    out, notes = {}, {}
    for key, fn, ds in (("crz", crz_entries, cfg.crz_entries), ("bus", bus_speeds, cfg.bus_speeds),
                        ("taxi_speeds", taxi_speeds, cfg.taxi_speeds)):  # fmt: skip
        try:
            out[key] = fn(src, ds)
            notes[key] = "ok"
        except Exception as exc:  # each MTA dataset is optional; the report says which ones are missing
            log.warning("MTA dataset %s (%s) unavailable: %s", key, ds, exc)
            notes[key] = f"unavailable: {str(exc)[:200]}"
    try:
        out["cbd_rows"] = src.socrata(cfg.cbd_zones, {"$order": ":id"}, max_rows=5000)
    except Exception as exc:
        log.warning("MTA CBD taxi zone list unavailable: %s", exc)
        out["cbd_rows"] = []
    out["notes"] = notes
    return out
