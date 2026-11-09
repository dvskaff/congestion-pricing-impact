"""SYNTHETIC data in the shapes of the real sources (TLC trip Parquet files, taxi zone lookup and shapefile, MTA
datasets on data.ny.gov), used only by the tests and the offline CI run. Pages built from it carry a red banner.

Known effects are planted so the tests can check the analysis recovers them: from January 2025, trips inside the
toll zone get 8% faster and pickups there fall 10%; bus segments inside the zone get 5% faster.
"""

from __future__ import annotations

import io
import re
import zipfile
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FALLBACK_CBD_ZONES, Config

START, END = "2023-01", "2025-06"
CBD = sorted(FALLBACK_CBD_ZONES)[:16]
UPPER = [140, 141, 142, 143, 151, 166, 236, 237, 238, 239, 262, 263]
OUTER = {"Brooklyn": [25, 33, 49, 65, 66, 97, 181], "Queens": [7, 82, 129, 260], "Bronx": [69, 159]}
OTHER = [(43, "Manhattan", "Central Park", "Yellow Zone"), (132, "Queens", "JFK Airport", "Airports"),
         (1, "EWR", "Newark Airport", "EWR")]  # fmt: skip
PLANTED_SPEED, PLANTED_PICKUPS, PLANTED_BUS = 0.08, -0.10, 0.05
FEE = {"yellow": 0.75, "green": 0.75, "fhvhv": 1.5}
TRIPS = {"yellow": 1.0, "green": 0.15, "fhvhv": 2.0}  # relative volume per service


def lookup() -> pd.DataFrame:
    rows = [(z, "Manhattan", f"Midtown {z}", "Yellow Zone") for z in CBD]
    rows += [(z, "Manhattan", f"Upper Manhattan {z}", "Boro Zone") for z in UPPER]
    rows += [(z, b, f"{b} {z}", "Boro Zone") for b, zs in OUTER.items() for z in zs]
    rows += OTHER
    return pd.DataFrame(rows, columns=["LocationID", "Borough", "Zone", "service_zone"])


def region_of(z: int) -> str:
    if z in CBD:
        return "cbd"
    if z in UPPER:
        return "upper"
    return "outer" if any(z in zs for zs in OUTER.values()) else "other"


def shapes_zip() -> bytes:
    """A real shapefile (squares on a grid, New York State Plane feet) with one zone split into two parts."""
    import shapefile

    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON)
    w.field("OBJECTID", "N")
    w.field("zone", "C")
    w.field("LocationID", "N")
    w.field("borough", "C")
    lk = lookup()
    for i, r in enumerate(lk.itertuples(index=False)):
        col, row = i % 8, i // 8
        x0, y0 = 975000 + col * 4000, 185000 + row * 4000
        sq = [[x0, y0], [x0, y0 + 3800], [x0 + 3800, y0 + 3800], [x0 + 3800, y0], [x0, y0]]  # clockwise (shapefile)
        w.poly([sq])
        w.record(i + 1, r.Zone, r.LocationID, r.Borough)
    w.poly([[[960000, 170000], [960000, 171000], [961000, 171000], [961000, 170000], [960000, 170000]]])
    w.record(len(lk) + 1, lk.iloc[0].Zone, int(lk.iloc[0].LocationID), "Manhattan")  # second part of zone 1's id
    w.close()
    prj = ('PROJCS["NAD83 / New York Long Island (ftUS)",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
           'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
           'PROJECTION["Lambert_Conformal_Conic_2SP"],UNIT["US survey foot",0.304800609601219]]')  # fmt: skip
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for ext, buf in (("shp", shp), ("shx", shx), ("dbf", dbf)):
            z.writestr(f"taxi_zones/taxi_zones.{ext}", buf.getvalue())
        z.writestr("taxi_zones/taxi_zones.prj", prj)
    return out.getvalue()


def _month_trips(service: str, month: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = pd.Period(month, "M")
    post = p >= pd.Period("2025-01", "M")
    season = 1 + 0.05 * np.sin(2 * np.pi * (p.month - 3) / 12)
    zones = CBD + UPPER + [z for zs in OUTER.values() for z in zs]
    zone_fx = {z: 1 + 0.15 * np.sin(z) for z in zones}
    n_day = int(110 * TRIPS[service])  # trips starting in each zone per day (scaled below)
    rows = []
    days = pd.date_range(p.start_time, p.end_time.normalize(), freq="D")
    for z in zones:
        reg = region_of(z)
        mult = (1 + PLANTED_PICKUPS) if (post and reg == "cbd") else 1.0
        n = rng.poisson(n_day * len(days) / 30 * mult * season * zone_fx[z] / 4)
        same = [x for x in zones if region_of(x) == reg]
        other = [x for x in zones if region_of(x) != reg]
        dz = np.where(rng.random(n) < 0.7, rng.choice(same, n), rng.choice(other, n))
        day = rng.choice(days, n)
        w = np.r_[[1.0] * 5, [3.0] * 16, [2.0] * 3]
        hour = rng.choice(24, n, p=w / w.sum())
        ts = pd.to_datetime(day) + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, n), unit="s")
        base = {"cbd": 8.0, "upper": 11.0, "outer": 13.0}[reg] * season * zone_fx[z]
        same_region = np.array([region_of(x) == reg for x in dz], dtype=bool)
        speed = base * np.where(same_region & post & (reg == "cbd"), 1 + PLANTED_SPEED, 1.0)
        speed = speed * np.exp(rng.normal(0, 0.15, n))
        miles = np.clip(rng.gamma(2.0, 1.2, n), 0.2, 30)
        mins = miles / speed * 60
        cbd_trip = (reg == "cbd") | np.isin(dz, CBD)
        fee = np.where(post & (cbd_trip | (rng.random(n) < 0.04)), FEE[service], 0.0)
        rows.append(pd.DataFrame({"ts": ts, "mins": mins, "miles": miles, "pu": z, "do": dz, "fee": fee}))
    df = pd.concat(rows, ignore_index=True)
    junk = df.sample(min(20, len(df)), random_state=seed).copy()  # rows the cleaning must drop
    junk["pu"] = 264
    late = df.sample(min(5, len(df)), random_state=seed + 1).copy()
    late["ts"] = p.start_time - pd.Timedelta(days=40)
    crazy = df.sample(min(10, len(df)), random_state=seed + 2).copy()
    crazy["mins"] = 0.1
    return pd.concat([df, junk, late, crazy], ignore_index=True)


def tlc_file(service: str, month: str, dest: Path) -> Path:
    t = _month_trips(service, month, seed=zlib.crc32(f"{service}{month}".encode()))
    drop = t["ts"] + pd.to_timedelta(t["mins"], unit="m")
    post = pd.Period(month, "M") >= pd.Period("2025-01", "M")
    if service == "fhvhv":
        df = pd.DataFrame({"hvfhs_license_num": "HV0003", "pickup_datetime": t["ts"], "dropoff_datetime": drop,
                           "PULocationID": t["pu"].astype("int32"), "DOLocationID": t["do"].astype("int32"),
                           "trip_miles": t["miles"], "trip_time": (t["mins"] * 60).round().astype("int64"),
                           "base_passenger_fare": 15.0})  # fmt: skip
    else:
        pre = "tpep" if service == "yellow" else "lpep"
        df = pd.DataFrame({"VendorID": 2, f"{pre}_pickup_datetime": t["ts"], f"{pre}_dropoff_datetime": drop,
                           "trip_distance": t["miles"], "PULocationID": t["pu"].astype("int64"),
                           "DOLocationID": t["do"].astype("int64"), "fare_amount": 12.5,
                           "Airport_fee": 0.0})  # fmt: skip
    if post:
        df["cbd_congestion_fee"] = t["fee"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


# --------------------------------------------------------------------------- MTA datasets (data.ny.gov)
def _crz() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    hours = pd.date_range("2025-01-05", "2026-03-31 23:00", freq="h")
    classes = {"1 - Cars, Pickups and Vans": 3000, "TLC Taxi/FHV": 1200, "2 - Single-Unit Trucks": 150}
    groups = ["Lincoln Tunnel", "Queensboro Bridge", "Brooklyn Bridge"]
    rows = []
    for c, base in classes.items():
        for g in groups:
            trend = np.where(hours.year == 2026, 0.95, 1.0)
            peak = np.where((hours.dayofweek < 5) & (hours.hour >= 5) & (hours.hour < 21), 1.0, 0.35)
            n = rng.poisson(base / 24 * peak * trend)
            rows.append(pd.DataFrame({"toll_date": hours.normalize().strftime("%Y-%m-%dT00:00:00.000"),
                                      "toll_hour": hours.strftime("%Y-%m-%dT%H:00:00.000"), "vehicle_class": c,
                                      "detection_group": g, "time_period": np.where(peak == 1.0, "Peak", "Overnight"),
                                      "crz_entries": n, "excluded_roadway_entries": rng.poisson(5, len(hours))}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def _bus() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    months = pd.period_range(START, END, freq="M")
    rows = []
    for r in [f"M{i}" for i in range(1, 11)]:
        for rel in ("Inside CBD", "Outside CBD"):
            for m in months:
                base = 6.0 if rel == "Inside CBD" else 8.5
                eff = 1 + PLANTED_BUS if (rel == "Inside CBD" and m >= pd.Period("2025-01", "M")) else 1.0
                rows.append({"month": m.start_time.strftime("%Y-%m-%dT00:00:00.000"), "route_id": r, "cbd_relation": rel,
                             "average_road_speed": base * eff * (1 + 0.03 * np.sin(m.month)) * np.exp(rng.normal(0, 0.02)),
                             "total_trips": int(rng.integers(500, 900))})  # fmt: skip
    return pd.DataFrame(rows)


def _taxi_speeds() -> pd.DataFrame:
    months = pd.period_range(START, END, freq="M")
    rows = []
    for m in months:
        for zone, base in (("CBD", 8.0), ("Adjacent to CBD", 10.0), ("Rest of NYC", 13.0)):
            eff = 1 + PLANTED_SPEED if (zone == "CBD" and m >= pd.Period("2025-01", "M")) else 1.0
            rows.append({"month": m.start_time.strftime("%Y-%m-%dT00:00:00.000"), "zone": zone,
                         "zonal_speed": base * eff * (1 + 0.05 * np.sin(2 * np.pi * (m.month - 3) / 12))})  # fmt: skip
    return pd.DataFrame(rows)


def _soql(df: pd.DataFrame, params: dict) -> list[dict]:
    """Enough of SoQL for the queries the project sends: field lists, sum/avg/count with aliases, group by."""
    sel = params.get("$select")
    if not sel:
        return df.head(int(params.get("$limit", len(df)))).astype(str).to_dict(orient="records")
    if int(params.get("$offset", 0)) > 0:
        return []
    group = [g.strip() for g in params.get("$group", "").split(",") if g.strip()]
    d = df.copy()
    aggs = {}
    for item in [s.strip() for s in sel.split(",")]:
        m = re.match(r"(sum|avg|count)\((.+)\)\s+AS\s+(\w+)", item, re.I)
        if not m:
            continue
        fn, expr, alias = m.group(1).lower(), m.group(2).strip(), m.group(3)
        if "*" in expr and expr != "*":
            a, b = (x.strip() for x in expr.split("*"))
            d[alias] = d[a].astype(float) * d[b].astype(float)
        elif expr == "*":
            d[alias] = 1
        else:
            d[alias] = d[expr].astype(float)
        aggs[alias] = {"sum": "sum", "avg": "mean", "count": "sum"}[fn]
    out = d.groupby(group, as_index=False).agg(aggs) if group else d.agg(aggs).to_frame().T
    return out.astype(str).to_dict(orient="records")


class FixtureSource:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tables = {}

    def today(self) -> date:
        return date(2025, 8, 15)

    def tlc_exists(self, service: str, month: str) -> bool:
        return START <= month <= END

    def latest_month(self, service: str = "yellow", today=None) -> str:
        return END

    def download_tlc(self, service: str, month: str, dest: Path) -> Path | None:
        return tlc_file(service, month, dest) if self.tlc_exists(service, month) else None

    def zone_lookup(self) -> pd.DataFrame:
        return lookup()

    def zone_shapes(self) -> bytes:
        return shapes_zip()

    def _table(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._tables:
            c = self.cfg
            make = {c.crz_entries: _crz, c.bus_speeds: _bus, c.taxi_speeds: _taxi_speeds,
                    c.cbd_zones: lambda: pd.DataFrame({"location_id": [str(z) for z in CBD],
                                                       "zone": [f"Midtown {z}" for z in CBD], "borough": "Manhattan"})}  # fmt: skip
            self._tables[dataset] = make[dataset]()
        return self._tables[dataset]

    def socrata(self, dataset: str, params: dict | None = None, page: int = 50_000, max_rows: int = 2_000_000):
        return _soql(self._table(dataset), params or {})
