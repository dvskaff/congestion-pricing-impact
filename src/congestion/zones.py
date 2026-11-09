"""TLC taxi zones: names, boroughs, which zones are inside the Congestion Relief Zone, analysis regions, and the
zone polygons (shapefile in New York State Plane feet -> GeoJSON in longitude/latitude) for the maps.

Which zones count as "inside the zone" is decided three ways, and the report shows how they agree:
1. the MTA's own list (data.ny.gov yfdc-w5jh);
2. the trip records themselves: a trip that starts and ends in the same taxi zone is charged the congestion fee only
   if that zone is inside the toll zone, so zones where most such trips since January 2025 paid the fee are inside;
3. a fixed list of zones south of 60th Street (fallback).
"""

from __future__ import annotations

import io
import json
import logging
import math
import zipfile

import numpy as np
import pandas as pd

from .config import FALLBACK_CBD_ZONES

log = logging.getLogger(__name__)
ID_KEYS = ("locationid", "location_id", "zone_id", "taxi_zone_id", "taxizone_id", "objectid")


# --------------------------------------------------------------------------- lookup and regions
def clean_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in lookup.columns}
    z = pd.DataFrame({
        "zone": pd.to_numeric(lookup[cols["locationid"]], errors="coerce"),
        "borough": lookup[cols["borough"]].astype(str).str.strip(),
        "zone_name": lookup[cols["zone"]].astype(str).str.strip(),
        "service_zone": lookup[cols.get("service_zone", cols["borough"])].astype(str).str.strip(),
    })  # fmt: skip
    return z.dropna(subset=["zone"]).astype({"zone": int}).query("zone >= 1 and zone <= 263").drop_duplicates("zone")


def regions(z: pd.DataFrame, cbd: set[int]) -> pd.DataFrame:
    """Analysis region of every zone. Controls exclude airports, parks and islands without regular street traffic."""
    z = z.copy()
    z["in_zone"] = z["zone"].isin(cbd)
    skip = z["service_zone"].str.contains("Airport", case=False) | z["borough"].isin(["EWR", "Unknown", "N/A", "nan"])
    skip |= z["zone_name"].str.contains(r"Park$|Island|Airport|Cemetery", case=False, regex=True) & ~z["in_zone"]
    z["region"] = np.select(
        [z["in_zone"], skip, z["borough"].eq("Manhattan"), z["borough"].isin(["Brooklyn", "Queens", "Bronx"])],
        ["Inside the toll zone", "Excluded", "Manhattan above 60th St", "Brooklyn, Queens and the Bronx"],
        default="Excluded",
    )
    return z


# --------------------------------------------------------------------------- which zones are inside
def official_cbd(rows: list[dict]) -> set[int]:
    """Taxi zone IDs from the MTA's CBD taxi zone dataset (field names discovered, not assumed)."""
    if not rows:
        return set()
    keys = {k.lower(): k for k in rows[0]}
    key = next((keys[k] for k in ID_KEYS if k in keys), None)
    if key is None:  # a geometry-only export: look for any integer-valued field named like an id
        key = next((v for k, v in keys.items() if "location" in k or k.endswith("_id")), None)
    if key is None:
        return set()
    ids = set()
    for r in rows:
        flag = next((str(v).lower() for k, v in r.items() if "cbd" in k.lower() and k != key), "true")
        try:
            if flag not in ("false", "0", "no", "n"):
                ids.add(int(float(r[key])))
        except (TypeError, ValueError):
            continue
    return {i for i in ids if 1 <= i <= 263}


def fee_cbd(con, min_trips: int = 50) -> tuple[set[int], pd.DataFrame]:
    """Zones where most trips starting and ending in the same zone since January 2025 paid the per-trip charge."""
    df = con.execute("""
        SELECT pu AS zone, sum(trips) AS trips, sum(fee_trips) AS fee_trips
        FROM od WHERE pu = "do" AND month >= DATE '2025-01-01' GROUP BY pu
    """).df()  # fmt: skip
    if df.empty:
        return set(), df
    df["fee_share"] = df["fee_trips"] / df["trips"]
    inside = set(df.loc[(df["trips"] >= min_trips) & (df["fee_share"] >= 0.5), "zone"].astype(int))
    return inside, df


def choose_cbd(official: set[int], derived: set[int]) -> tuple[set[int], dict]:
    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else None

    info = {"official": sorted(official), "from_fees": sorted(derived), "fallback": sorted(FALLBACK_CBD_ZONES),
            "agreement_official_fees": jac(official, derived), "agreement_fees_fallback": jac(derived, set(FALLBACK_CBD_ZONES)),
            "agreement_official_fallback": jac(official, set(FALLBACK_CBD_ZONES))}  # fmt: skip
    if 10 <= len(official) <= 90:
        info["used"] = "MTA list (data.ny.gov)"
        return official, info
    if 10 <= len(derived) <= 90:
        info["used"] = "per-trip congestion fees in the trip records"
        return derived, info
    info["used"] = "fixed list of zones south of 60th Street"
    return set(FALLBACK_CBD_ZONES), info


# --------------------------------------------------------------------------- polygons
def _dp(points, tol):
    if len(points) < 5:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = points[i], points[j]
        dx, dy = x2 - x1, y2 - y1
        best, idx = 0.0, None
        for k in range(i + 1, j):
            x, y = points[k]
            if dx == dy == 0:
                d = math.hypot(x - x1, y - y1)
            else:
                t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
                d = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
            if d > best:
                best, idx = d, k
        if idx is not None and best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    out = [p for p, k in zip(points, keep, strict=True) if k]
    return out if len(out) >= 4 else points


def _area(ring) -> float:
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:], strict=False)) / 2


def _polys(geom: dict) -> list:
    return [geom["coordinates"]] if geom["type"] == "Polygon" else list(geom["coordinates"])


def shapes_to_geojson(zip_bytes: bytes, tol: float = 0.00008) -> dict:
    """TLC taxi_zones.zip (shapefile, EPSG:2263) -> FeatureCollection in lon/lat, one feature per zone, simplified to
    ~8 m, rings in the GeoJSON order (outer counter-clockwise)."""
    import shapefile
    from pyproj import Transformer

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = {n.lower().rsplit("/", 1)[-1]: n for n in zf.namelist()}
    base = next(n[:-4] for n in names if n.endswith(".shp"))
    rd = shapefile.Reader(shp=io.BytesIO(zf.read(names[base + ".shp"])),
                          dbf=io.BytesIO(zf.read(names[base + ".dbf"])),
                          shx=io.BytesIO(zf.read(names[base + ".shx"])) if base + ".shx" in names else None)  # fmt: skip
    prj = zf.read(names[base + ".prj"]).decode() if base + ".prj" in names else ""
    projected = "PROJCS" in prj or "PROJCRS" in prj
    tr = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True) if projected else None
    fields = [f[0].lower() for f in rd.fields[1:]]
    id_field = next(i for i, f in enumerate(fields) if f in ("locationid", "location_i", "location_id"))
    merged: dict[int, list] = {}
    for sr in rd.iterShapeRecords():
        zid = int(sr.record[id_field])
        geom = sr.shape.__geo_interface__
        for poly in _polys(geom):
            rings = []
            for ring in poly:
                if tr:
                    xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
                    ring = list(zip(xs, ys, strict=True))
                ring = [[round(x, 5), round(y, 5)] for x, y in _dp([tuple(p[:2]) for p in ring], tol)]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(ring)
            if rings:
                outer, holes = rings[0], rings[1:]
                outer = outer if _area(outer) > 0 else outer[::-1]
                holes = [h if _area(h) < 0 else h[::-1] for h in holes]
                merged.setdefault(zid, []).append([outer, *holes])
    feats = [{"type": "Feature", "id": z, "properties": {"zone": z},
              "geometry": {"type": "MultiPolygon", "coordinates": polys}} for z, polys in sorted(merged.items())]  # fmt: skip
    return {"type": "FeatureCollection", "features": feats}


def for_d3(fc: dict) -> dict:
    """Plotly's geo maps (d3) need the opposite ring order: outer clockwise."""
    out = []
    for f in fc["features"]:
        polys = [[ring[::-1] for ring in poly] for poly in f["geometry"]["coordinates"]]
        out.append({**f, "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    return {"type": "FeatureCollection", "features": out}


def save_geojson(fc: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, separators=(",", ":")))
