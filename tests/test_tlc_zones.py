"""Trip-record aggregation, zone polygons and the definition of the toll zone."""

import duckdb
import pandas as pd
import pytest

from congestion import fixture, tlc, zones
from congestion.config import FALLBACK_CBD_ZONES


@pytest.mark.parametrize("service", ["yellow", "green", "fhvhv"])
@pytest.mark.parametrize("month", ["2024-11", "2025-02"])
def test_aggregate_handles_each_layout(tmp_path, service, month):
    raw = fixture.tlc_file(service, month, tmp_path / "raw.parquet")
    info = tlc.aggregate(raw, service, month, tmp_path / "agg")
    assert info["has_fee_column"] == (month >= "2025-01")
    assert info["rows_read"] - info["rows_kept"] >= 25  # 20 bad zone ids + 5 trips from another month dropped
    con = duckdb.connect()
    od = con.execute(f"SELECT * FROM read_parquet('{tmp_path}/agg/od/{service}_{month}.parquet')").df()
    zd = con.execute(f"SELECT * FROM read_parquet('{tmp_path}/agg/zoneday/{service}_{month}.parquet')").df()
    hr = con.execute(f"SELECT * FROM read_parquet('{tmp_path}/agg/hourly/{service}_{month}.parquet')").df()
    assert od["trips"].sum() == info["rows_kept"] == zd["pickups"].sum() == zd["dropoffs"].sum() == hr["trips"].sum()
    assert set(od["period"]) == {"weekday_peak", "weekend_peak", "overnight"}
    assert (od["speed_trips"] <= od["trips"]).all() and od["speed_trips"].sum() < od["trips"].sum()  # crazy speeds out
    assert pd.to_datetime(zd["date"]).dt.strftime("%Y-%m").eq(month).all()
    mph = od["miles"].sum() / od["hours"].sum()
    assert 5 < mph < 20
    charged = od["fee_trips"].sum()
    assert (charged > 0) == (month >= "2025-01")


def test_periods_follow_toll_hours(tmp_path):
    p = tmp_path / "y.parquet"
    ts = pd.to_datetime(
        [
            "2025-03-03 04:59",
            "2025-03-03 05:00",
            "2025-03-03 20:59",
            "2025-03-03 21:00",
            "2025-03-08 08:59",
            "2025-03-08 09:00",
            "2025-03-09 20:30",
        ]
    )  # Mon x4, Sat x2, Sun
    pd.DataFrame({"tpep_pickup_datetime": ts, "tpep_dropoff_datetime": ts + pd.Timedelta(minutes=10),
                  "trip_distance": 1.5, "PULocationID": 161, "DOLocationID": 161}).to_parquet(p)  # fmt: skip
    tlc.aggregate(p, "yellow", "2025-03", tmp_path / "agg")
    od = duckdb.connect().execute(f"SELECT period, trips FROM '{tmp_path}/agg/od/yellow_2025-03.parquet'").df()
    got = dict(zip(od["period"], od["trips"], strict=True))
    assert got == {"overnight": 3, "weekday_peak": 2, "weekend_peak": 2}


def test_unexpected_layout_is_rejected(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(p)
    with pytest.raises(ValueError, match="unexpected columns"):
        tlc.aggregate(p, "yellow", "2025-01", tmp_path / "agg")


def test_shapefile_to_geojson():
    fc = zones.shapes_to_geojson(fixture.shapes_zip())
    ids = [f["id"] for f in fc["features"]]
    assert len(ids) == len(set(ids)) == len(fixture.lookup())
    first = next(f for f in fc["features"] if f["id"] == int(fixture.lookup().iloc[0].LocationID))
    assert len(first["geometry"]["coordinates"]) == 2  # two shapefile records for one zone become one MultiPolygon
    for f in fc["features"]:
        for poly in f["geometry"]["coordinates"]:
            ring = poly[0]
            assert ring[0] == ring[-1] and zones._area(ring) > 0  # closed, counter-clockwise (GeoJSON order)
            assert all(-74.3 < x < -73.6 and 40.4 < y < 41.0 for x, y in ring)  # reprojected to New York lon/lat
    d3 = zones.for_d3(fc)
    assert zones._area(d3["features"][0]["geometry"]["coordinates"][0][0]) < 0


def test_official_list_and_choice():
    assert zones.official_cbd([{"location_id": "161", "zone": "Midtown Center"}, {"location_id": "4"}]) == {161, 4}
    assert zones.official_cbd([{"LocationID": 90.0, "cbd_flag": "Yes"}, {"LocationID": 41, "cbd_flag": "No"}]) == {90}
    assert zones.official_cbd([]) == set()
    many = set(range(100, 140))
    used, info = zones.choose_cbd(many, set())
    assert used == many and info["used"].startswith("MTA")
    used, info = zones.choose_cbd(set(), many)
    assert used == many and "fees" in info["used"]
    used, info = zones.choose_cbd({1, 2}, set())
    assert used == set(FALLBACK_CBD_ZONES) and info["agreement_official_fees"] == 0.0


def test_regions():
    z = zones.regions(zones.clean_lookup(fixture.lookup()), set(fixture.CBD))
    r = z.set_index("zone")["region"]
    assert r[fixture.CBD[0]] == "Inside the toll zone" and r[fixture.UPPER[0]] == "Manhattan above 60th St"
    assert r[25] == "Brooklyn, Queens and the Bronx"
    assert r[43] == r[132] == r[1] == "Excluded"  # a park, an airport, Newark
