"""The analysis: did the toll speed up traffic in the zone, and what happened to trips and vehicle entries?

Main design: difference-in-differences on a panel of taxi zones by month. For every zone and month, the average speed
of taxi and ride-hail trips that start in the zone and end in the same region (weekday toll hours), from TLC trip
records. Zones inside the toll zone are compared with zones in Manhattan above 60th Street before and after
January 2025, with zone and month fixed effects and standard errors clustered by zone:

    log(speed[z, t]) = b * inside[z] * after[t] + zone[z] + month[t] + e

exp(b) - 1 is the change in speed attributable to the toll under the parallel-trends assumption, which the event
study (one coefficient per month) and a placebo year (a fake toll in January 2024) put to the test.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import POLICY_MONTH

log = logging.getLogger(__name__)
INSIDE, UPPER, OUTER = "Inside the toll zone", "Manhattan above 60th St", "Brooklyn, Queens and the Bronx"
RELATIONS = ("Within the zone", "Into the zone", "Out of the zone", "Outside the zone")
SERVICE_NAMES = {"yellow": "Yellow taxi", "green": "Green taxi", "fhvhv": "Uber, Lyft and other app rides"}


def connect(cfg) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for kind in ("od", "zoneday", "hourly"):
        files = [f.as_posix() for f in sorted((cfg.agg_dir / kind).glob("*.parquet"))]
        con.execute(f"CREATE VIEW {kind} AS SELECT * FROM read_parquet({files})")
    return con


def set_zones(con, zones: pd.DataFrame) -> None:
    con.register("zones_df", zones[["zone", "borough", "zone_name", "region", "in_zone"]])
    con.execute("CREATE OR REPLACE TABLE zones AS SELECT * FROM zones_df")


def complete_months(con, services) -> list[pd.Timestamp]:
    """Months for which every service has been loaded (the newest month can lag for one of them)."""
    df = con.execute("SELECT month, count(DISTINCT service) AS s FROM od GROUP BY month ORDER BY month").df()
    return [pd.Timestamp(m) for m in df.loc[df["s"] >= len(services), "month"]]


# --------------------------------------------------------------------------- descriptive series
def relation_month(con) -> pd.DataFrame:
    """Trips by month, service, period and relation to the zone (within / into / out of / outside)."""
    return con.execute(f"""
        SELECT o.month, o.service, o.period,
               CASE WHEN a.in_zone AND b.in_zone THEN '{RELATIONS[0]}' WHEN b.in_zone THEN '{RELATIONS[1]}'
                    WHEN a.in_zone THEN '{RELATIONS[2]}' ELSE '{RELATIONS[3]}' END AS relation,
               sum(trips) AS trips, sum(fee_trips) AS fee_trips, sum(fee_sum) AS fee_sum,
               sum(speed_trips) AS speed_trips, sum(miles) AS miles, sum(hours) AS hours
        FROM od o JOIN zones a ON a.zone = o.pu JOIN zones b ON b.zone = o."do"
        GROUP BY ALL ORDER BY 1, 2, 3, 4
    """).df()  # fmt: skip


def region_speed_month(con, period: str = "weekday_peak") -> pd.DataFrame:
    """Average speed of trips that start and end in the same region, by month (all services together)."""
    return con.execute(f"""
        SELECT o.month, a.region, sum(miles) / sum(hours) AS mph, sum(speed_trips) AS trips
        FROM od o JOIN zones a ON a.zone = o.pu JOIN zones b ON b.zone = o."do"
        WHERE a.region = b.region AND a.region <> 'Excluded' AND o.period = '{period}'
        GROUP BY ALL ORDER BY 1, 2
    """).df()  # fmt: skip


def zone_panel(con, cfg, period: str = "weekday_peak", service: str | None = None) -> pd.DataFrame:
    """Zone x month: speed of trips from the zone to the same region, and pickups from the zone (all destinations)."""
    svc = f"AND o.service = '{service}'" if service else ""
    sp = con.execute(f"""
        SELECT o.pu AS zone, o.month, sum(miles) / sum(hours) AS mph, sum(speed_trips) AS speed_trips
        FROM od o JOIN zones a ON a.zone = o.pu JOIN zones b ON b.zone = o."do"
        WHERE a.region = b.region AND a.region <> 'Excluded' AND o.period = '{period}' {svc}
        GROUP BY ALL
    """).df()  # fmt: skip
    pk = con.execute(f"""
        SELECT o.pu AS zone, o.month, sum(trips) AS pickups FROM od o WHERE TRUE {svc} GROUP BY ALL
    """).df()  # fmt: skip
    df = sp.merge(pk, on=["zone", "month"], how="outer").merge(
        con.execute("SELECT zone, region, zone_name, borough FROM zones").df(), on="zone"
    )
    df = df[df["region"] != "Excluded"]
    df["month"] = pd.to_datetime(df["month"])
    ok = df["speed_trips"].fillna(0) >= cfg.min_zone_month_trips
    df.loc[~ok, ["mph"]] = np.nan
    n_months = df["month"].nunique()
    cover = df.dropna(subset=["mph"]).groupby("zone")["month"].nunique() / max(n_months, 1)
    keep = cover[cover >= cfg.min_zone_coverage].index
    return df[df["zone"].isin(keep)].reset_index(drop=True)


# --------------------------------------------------------------------------- difference-in-differences
def _prep(panel, outcome, control, start=None, end=None, post=POLICY_MONTH):
    d = panel[panel["region"].isin([INSIDE, control])].copy()
    if start:
        d = d[d["month"] >= pd.Timestamp(start)]
    if end:
        d = d[d["month"] <= pd.Timestamp(end)]
    d = d[d[outcome] > 0].dropna(subset=[outcome])
    d["y"] = np.log(d[outcome])
    d["treated"] = (d["region"] == INSIDE).astype(int)
    d["post"] = (d["month"] >= pd.Timestamp(post)).astype(int)
    d["did"] = d["treated"] * d["post"]
    d["m"] = d["month"].dt.strftime("%Y_%m")
    d["cal"] = d["month"].dt.month
    return d


BASE_YEAR = pd.Timestamp(POLICY_MONTH).year - 1  # the last full year before the toll


def _fit(d: pd.DataFrame, terms: list[str], weight: str | None = None):
    f = "y ~ " + " + ".join(terms) + " + C(unit) + C(m)"
    model = smf.wls(f, d, weights=d[weight]) if weight else smf.ols(f, d)
    return model.fit(cov_type="cluster", cov_kwds={"groups": d["cluster"]})


def _est(r, name: str, label: str, d: pd.DataFrame, **extra) -> dict:
    b, se = float(r.params[name]), float(r.bse[name])
    return {"label": label, "ok": True, "coef": b, "se": se, "p": float(r.pvalues[name]),
            "effect": float(np.exp(b) - 1), "lo": float(np.exp(b - 1.96 * se) - 1), "hi": float(np.exp(b + 1.96 * se) - 1),
            "zones_inside": int(d.loc[d.treated == 1, "unit"].nunique()),
            "zones_control": int(d.loc[d.treated == 0, "unit"].nunique()), "obs": int(len(d)), **extra}  # fmt: skip


def frame(panel, outcome="mph", control=UPPER) -> pd.DataFrame:
    d = _prep(panel, outcome, control)
    d["unit"] = d["cluster"] = d["zone"]
    return d


def yearly(d: pd.DataFrame, weight: str | None = None) -> dict[int, dict]:
    """Inside-vs-control gap in each calendar year relative to the last pre-toll year (2024), with zone and month
    effects. The year before (2023) is a built-in placebo: no toll yet, so it should be about zero."""
    d = d.copy()
    years = [y for y in sorted(d["month"].dt.year.unique()) if y != BASE_YEAR]
    if BASE_YEAR not in set(d["month"].dt.year) or d["treated"].nunique() < 2 or not years:
        return {}
    for y in years:
        d[f"t{y}"] = d["treated"] * (d["month"].dt.year == y)
    r = _fit(d, [f"t{y}" for y in years], weight)
    return {int(y): _est(r, f"t{y}", f"{y} vs. {BASE_YEAR}", d, year=int(y)) for y in years}


def trend_adjusted(d: pd.DataFrame, label: str) -> dict:
    """Difference-in-differences that lets the inside zones follow their own linear trend: the toll's effect is the
    average jump from that trend after January 2025. Right if the pre-toll drift would have continued."""
    d = d.copy()
    d["tt"] = d["treated"] * (d["month"] - pd.Timestamp(POLICY_MONTH)).dt.days / 365.25
    r = _fit(d, ["did", "tt"])
    return _est(r, "did", label, d, trend_per_year=float(np.exp(r.params["tt"]) - 1))


def pooled(d: pd.DataFrame, label: str) -> dict:
    """The textbook version: all toll months vs. all earlier months. Valid only if trends were parallel."""
    return _est(_fit(d, ["did"]), "did", label, d)


def did(panel, outcome="mph", control=UPPER, label="") -> dict:
    d = frame(panel, outcome, control)
    if d["treated"].nunique() < 2 or d["post"].nunique() < 2:
        return {"label": label, "ok": False}
    return pooled(d, label)


def event_study(panel, outcome="mph", control=UPPER) -> tuple[pd.DataFrame, dict]:
    """One coefficient per month: the inside-vs-control gap relative to its average in 2024 (all twelve months of the
    last pre-toll year form the reference, so one unusual month can't tilt the picture). Also returns the pre-toll
    trend test."""
    d = frame(panel, outcome, control)
    names = []
    for mo in sorted(d["month"].unique()):
        if pd.Timestamp(mo).year == BASE_YEAR:
            continue
        n = "ev_" + pd.Timestamp(mo).strftime("%Y_%m")
        d[n] = ((d["month"] == mo) & (d["treated"] == 1)).astype(int)
        names.append(n)
    r = _fit(d, names)
    rows = []
    for n in names:
        b, se = r.params[n], r.bse[n]
        rows.append({"month": pd.Timestamp(n[3:].replace("_", "-") + "-01"), "coef": b, "effect": np.exp(b) - 1,
                     "lo": np.exp(b - 1.96 * se) - 1, "hi": np.exp(b + 1.96 * se) - 1})  # fmt: skip
    for mo in sorted(d.loc[d["month"].dt.year == BASE_YEAR, "month"].unique()):
        rows.append({"month": pd.Timestamp(mo), "coef": 0.0, "effect": 0.0, "lo": np.nan, "hi": np.nan})
    es = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    return es, pre_trend(d)


def pre_trend(d: pd.DataFrame) -> dict:
    """Before the toll, did the inside zones already drift away from the control zones? A separate linear trend for
    the inside zones, fitted on pre-toll months only (one slope stays reliable with a few dozen clustered zones,
    unlike a joint test of many monthly coefficients)."""
    pre = d[d["post"] == 0].copy()
    if pre["treated"].nunique() < 2 or pre["month"].nunique() < 6:
        return {"slope_per_year": np.nan, "p": np.nan}
    pre["tt"] = pre["treated"] * (pre["month"] - pre["month"].min()).dt.days / 365.25
    r = _fit(pre, ["tt"])
    return {"slope_per_year": float(np.exp(r.params["tt"]) - 1), "p": float(r.pvalues["tt"]),
            "months": int(pre["month"].nunique())}  # fmt: skip


def effects(d: pd.DataFrame, what: str = "") -> dict:
    """The estimates reported for one outcome: each year vs. 2024, trend-adjusted, and pooled."""
    yr = yearly(d)
    first = int(pd.Timestamp(POLICY_MONTH).year)
    out = {"yearly": yr,
           "trend": trend_adjusted(d, f"{what}Trend-adjusted: if the pre-toll drift had continued"),
           "pooled": pooled(d, f"{what}Pooled: all toll months vs. all earlier months (assumes parallel trends)")}  # fmt: skip
    for key, y, text in (("first", first, "First toll year"), ("second", first + 1, "Second toll year"),
                         ("placebo", BASE_YEAR - 1, "Placebo, before any toll")):  # fmt: skip
        if y in yr:
            suffix = " (should be about zero)" if key == "placebo" else ""
            out[key] = {**yr[y], "label": f"{what}{text}: {y} vs. {BASE_YEAR}{suffix}"}
    return out


def speed_results(panels: dict) -> dict:
    """Speeds: the main estimates and robustness checks (each robustness row is the first toll year vs. 2024)."""
    main = effects(frame(panels["all"]))
    first = int(pd.Timestamp(POLICY_MONTH).year)
    robust = []
    checks = [("all", OUTER, None, "Control: Brooklyn, Queens and the Bronx"),
              ("all", UPPER, "speed_trips", "Weighted by number of trips"),
              ("weekend_peak", UPPER, None, "Weekend toll hours"), ("overnight", UPPER, None, "Overnight (lower toll)"),
              ("yellow", UPPER, None, "Yellow taxis only"), ("fhvhv", UPPER, None, "Uber, Lyft and other app rides only")]  # fmt: skip
    for key, control, weight, label in checks:
        p = panels.get(key)
        if p is None or p.empty:
            continue
        d = frame(p, control=control)
        if d["treated"].nunique() < 2:
            continue
        yr = yearly(d, weight=weight).get(first)
        if yr:
            robust.append({**yr, "label": f"{label}, {first} vs. {BASE_YEAR}"})
    main["robust"] = robust
    return main


def ordered(res: dict) -> list[dict]:
    """Estimates in display order."""
    rows = [res.get(k) for k in ("first", "second", "trend", "placebo", "pooled")]
    return [r for r in rows if r] + res.get("robust", [])


# --------------------------------------------------------------------------- vehicle entries (MTA)
def crz_summary(crz: dict) -> dict:
    d = crz["daily"].dropna(subset=["date"])
    total = d.groupby("date")["entries"].sum().rename("entries").reset_index()
    total["weekday"] = total["date"].dt.dayofweek < 5
    last = total["date"].max()
    out = {"first_day": str(total["date"].min().date()), "last_day": str(last.date()), "daily": total,
           "per_weekday": float(total.loc[total.weekday, "entries"].mean())}  # fmt: skip
    years = sorted(total["date"].dt.year.unique())
    if len(years) >= 2 and last.year == years[-1]:
        y1, y0 = years[-1], years[-2]
        end1, end0 = last, last - pd.DateOffset(years=1)

        def window(dates, y, end):  # the same calendar window (Jan 5 to the latest day) in a given year
            return (dates >= pd.Timestamp(f"{y}-01-05")) & (dates <= end)

        cur, prev = total[window(total["date"], y1, end1)], total[window(total["date"], y0, end0)]
        out["yoy"] = {"year": int(y1), "prior": int(y0), "through": f"{last:%b %d}",
                      "change": float(cur["entries"].sum() / prev["entries"].sum() - 1),
                      "per_weekday": float(cur.loc[cur.weekday, "entries"].mean()),
                      "per_weekday_prior": float(prev.loc[prev.weekday, "entries"].mean())}  # fmt: skip
        g = d[window(d["date"], y1, end1) | window(d["date"], y0, end0)].assign(y=lambda x: x["date"].dt.year)
        for col, key in (("vehicle_class", "by_class"), ("entry_point", "by_entry")):
            if col in g:
                p = g.pivot_table(index=col, columns="y", values="entries", aggfunc="sum").fillna(0)
                p = p.rename(columns={y0: "prior", y1: "current"})
                p["change"] = p["current"] / p["prior"].replace(0, np.nan) - 1
                p["share"] = p["current"] / p["current"].sum()
                out[key] = p.sort_values("current", ascending=False).reset_index().rename(columns={col: "name"})
    if "period" in d:
        out["period_share"] = (d.groupby("period")["entries"].sum() / d["entries"].sum()).to_dict()
    if "hourly" in crz and len(crz["hourly"]):
        h = crz["hourly"].copy()
        h["hour"], h["weekday"], h["year"] = h["ts"].dt.hour, h["ts"].dt.dayofweek < 5, h["ts"].dt.year
        days = h.assign(day=h["ts"].dt.normalize()).groupby(["year", "weekday"])["day"].nunique()
        prof = h.groupby(["year", "weekday", "hour"])["entries"].sum().reset_index()
        prof["per_day"] = prof["entries"] / [days[(y, w)] for y, w in zip(prof["year"], prof["weekday"], strict=True)]
        out["hour_profile"] = prof
    return out


# --------------------------------------------------------------------------- bus speeds (MTA)
def bus_did(bus: pd.DataFrame) -> dict | None:
    """The same design on MTA bus speeds: route segments inside the zone vs. outside it."""
    from .mta import relation_role

    b = bus.copy()
    b["role"] = b["relation"].map(relation_role)
    b = b.dropna(subset=["role"])
    if b["role"].nunique() < 2 or b["month"].dt.year.min() > BASE_YEAR:
        return None
    b["unit"] = b["route"].astype(str) + " | " + b["relation"].astype(str)
    b["cluster"] = b["route"].astype(str)
    b = b[b["speed"] > 0]
    cover = b.groupby("unit")["month"].nunique() / b["month"].nunique()
    b = b[b["unit"].isin(cover[cover >= 0.8].index)].copy()
    b["y"], b["treated"] = np.log(b["speed"]), (b["role"] == "treated").astype(int)
    b["post"] = (b["month"] >= pd.Timestamp(POLICY_MONTH)).astype(int)
    b["did"] = b["treated"] * b["post"]
    b["m"] = b["month"].dt.strftime("%Y_%m")
    out = effects(b, "Buses: ")
    out["series"] = b.groupby(["month", "role"])["speed"].mean().unstack().reset_index()
    out["routes"], out["units"] = int(b["route"].nunique()), int(b["unit"].nunique())
    out["relations"] = sorted(map(str, bus["relation"].unique()))
    return out


# --------------------------------------------------------------------------- per-zone changes (map)
def zone_changes(con, panel: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    """For each zone: change in internal-trip speed and in pickups, toll months vs. the same months a year earlier,
    and the share of its same-zone trips charged the fee since January 2025."""
    post = [m for m in months if m >= pd.Timestamp(POLICY_MONTH)][-12:]
    pre = [m - pd.DateOffset(years=1) for m in post]
    rows = []
    for z, g in panel.groupby("zone"):
        g = g.set_index("month")
        a, b = g.reindex(post), g.reindex(pre)
        sp_a = (a["mph"] * a["speed_trips"]).sum() / a.loc[a["mph"].notna(), "speed_trips"].sum()
        sp_b = (b["mph"] * b["speed_trips"]).sum() / b.loc[b["mph"].notna(), "speed_trips"].sum()
        rows.append({"zone": z, "speed_change": sp_a / sp_b - 1 if sp_b else np.nan,
                     "pickups_change": a["pickups"].sum() / b["pickups"].sum() - 1 if b["pickups"].sum() else np.nan,
                     "mph_after": sp_a})  # fmt: skip
    out = pd.DataFrame(rows)
    fee = con.execute("""
        SELECT pu AS zone, sum(fee_trips) / sum(trips) AS fee_share FROM od
        WHERE month >= DATE '2025-01-01' AND pu = "do" GROUP BY pu
    """).df()  # fmt: skip
    z = con.execute("SELECT zone, zone_name, borough, region, in_zone FROM zones").df()
    return z.merge(out, on="zone", how="left").merge(fee, on="zone", how="left")


def fee_check(rel: pd.DataFrame) -> pd.DataFrame:
    r = rel[rel["month"] >= pd.Timestamp(POLICY_MONTH)].groupby(["relation", "service"])[["trips", "fee_trips"]].sum()
    r["share_charged"] = r["fee_trips"] / r["trips"]
    return r.reset_index()


def yoy(series: pd.DataFrame, value: str, months: list[pd.Timestamp], weight: str | None = None) -> float:
    """Toll months (up to the latest 12) vs. the same calendar months a year earlier."""
    post = [m for m in months if m >= pd.Timestamp(POLICY_MONTH)][-12:]
    pre = [m - pd.DateOffset(years=1) for m in post]
    s = series.set_index("month")
    if weight:
        a = (s.loc[s.index.isin(post), value] * s.loc[s.index.isin(post), weight]).sum() / s.loc[
            s.index.isin(post), weight
        ].sum()
        b = (s.loc[s.index.isin(pre), value] * s.loc[s.index.isin(pre), weight]).sum() / s.loc[
            s.index.isin(pre), weight
        ].sum()
    else:
        a, b = s.loc[s.index.isin(post), value].sum(), s.loc[s.index.isin(pre), value].sum()
    return float(a / b - 1) if b else np.nan
