"""MTA datasets: field discovery and the SoQL queries (against the fixture's tiny SoQL engine)."""

from congestion import fixture, mta


def test_find_matches_by_meaning():
    f = ["toll_date", "toll_week", "vehicle_class", "crz_entries", "excluded_roadway_entries"]
    assert mta.find(f, "toll date", "date", exclude=("week",)) == "toll_date"
    assert mta.find(f, "crz entries", "entries", exclude=("excluded",)) == "crz_entries"
    assert mta.find(f, "detection group") is None
    assert mta.find(["Month", "Zone", "Zonal Speed"], "zonal speed", "speed") == "Zonal Speed"


def test_relation_roles():
    assert mta.relation_role("Inside CBD") == "treated" and mta.relation_role("Outside CBD") == "control"
    assert mta.relation_role("CBD") == "treated" and mta.relation_role("Non-CBD") == "control"
    assert mta.relation_role("Crosses CBD") is None and mta.relation_role("Partially in CBD") is None


def test_datasets_load(cfg):
    src = fixture.FixtureSource(cfg)
    out = mta.load_all(src, cfg)
    assert out["notes"] == {"crz": "ok", "bus": "ok", "taxi_speeds": "ok"}
    d = out["crz"]["daily"]
    assert {"date", "vehicle_class", "entry_point", "period", "entries"} <= set(d.columns)
    assert d["entries"].sum() == out["crz"]["hourly"]["entries"].sum()
    b = out["bus"]
    assert set(b["relation"]) == {"Inside CBD", "Outside CBD"} and b["speed"].between(4, 12).all()
    assert len(out["cbd_rows"]) == len(fixture.CBD)


def test_missing_dataset_is_reported_not_fatal(cfg):
    class Broken(fixture.FixtureSource):
        def socrata(self, dataset, params=None, page=50_000, max_rows=2_000_000):
            if dataset == cfg.bus_speeds:
                raise RuntimeError("HTTP 404")
            return super().socrata(dataset, params, page, max_rows)

    out = mta.load_all(Broken(cfg), cfg)
    assert "bus" not in out and out["notes"]["bus"].startswith("unavailable") and "crz" in out
