"""The dashboard (site/index.html) and results.json."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version

from . import zones as Z
from .analysis import INSIDE, OUTER, RELATIONS, SERVICE_NAMES, UPPER, yoy
from .config import POLICY_MONTH, Config
from .report_base import (
    AQUA,
    BLUE,
    CSS,
    GRID,
    LAYOUT_JS,
    MUTED,
    ORANGE,
    SURFACE,
    VIOLET,
    _card,
    _div,
    _int,
    _style,
    _table,
)

REGION_COLORS = {INSIDE: ORANGE, UPPER: BLUE, OUTER: AQUA}
EXTRA_CSS = """
.cards { grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); }
main a { color:#2a78d6; }
.callout { border-left:4px solid #b7791f; background:#fbf6ea; padding:10px 14px; border-radius:6px; margin:10px 0 14px;
           color:#52514e; max-width:920px; font-size:14px; }
.ci { display:block; font-size:11.5px; color:#898781; }
.btn.dark { background:#104281; color:#fff; }
#zone-detail { min-height:330px; }
.zone-pick { font-size:13px; color:#52514e; margin:4px 0 0; }
"""


def _pp(v, d=1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _ci(r) -> str:
    return f"{_pp(r['effect'])}<span class='ci'>95% CI {_pp(r['lo'])} to {_pp(r['hi'])}</span>"


def _toll_line(fig, y0=0, y1=1):
    x = pd.Timestamp(POLICY_MONTH)
    fig.add_vline(x=x.timestamp() * 1000, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=x, y=1, yref="paper", text="Toll starts", showarrow=False, xanchor="left",
                       font=dict(size=11, color=MUTED), yshift=4)  # fmt: skip


# --------------------------------------------------------------------------- figures
def fig_speeds(sp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for reg in (INSIDE, UPPER, OUTER):
        d = sp[sp["region"] == reg].sort_values("month")
        fig.add_trace(go.Scatter(x=d["month"], y=d["mph"], name=reg, mode="lines+markers",
                                 line=dict(color=REGION_COLORS[reg], width=2.5 if reg == INSIDE else 2),
                                 marker=dict(size=5), hovertemplate="%{x|%b %Y}: %{y:.2f} mph"))  # fmt: skip
    _style(fig, "Average speed of trips within each area, weekday toll hours", height=400, y_title="mph",
           legend=dict(orientation="h", y=-0.15, x=0, yanchor="top"), margin=dict(l=60, r=20, t=60, b=80))  # fmt: skip
    _toll_line(fig)
    return fig


def fig_event(es: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    post = es["month"] >= pd.Timestamp(POLICY_MONTH)
    fig = go.Figure()
    for flag, color, name in ((False, MUTED, "Before the toll"), (True, ORANGE, "After")):
        d = es[post == flag]
        fig.add_trace(go.Scatter(
            x=d["month"], y=d["effect"], mode="markers", name=name, marker=dict(color=color, size=7),
            error_y=dict(type="data", symmetric=False, array=d["hi"] - d["effect"], arrayminus=d["effect"] - d["lo"],
                         color=color, thickness=1.2, width=0),
            hovertemplate="%{x|%b %Y}: %{y:+.1%}<extra></extra>"))  # fmt: skip
    fig.add_hline(y=0, line=dict(color=GRID, width=1.5))
    _style(fig, title, height=380, y_title=y_title, hovermode="closest",
           legend=dict(orientation="h", y=-0.15, x=0, yanchor="top"), margin=dict(l=60, r=20, t=60, b=80))  # fmt: skip
    fig.update_yaxes(tickformat="+.0%")
    _toll_line(fig)
    return fig


def fig_forest(results: list[dict]) -> go.Figure:
    labels = [r["label"] for r in results][::-1]
    eff = [r["effect"] for r in results][::-1]
    lo = [r["lo"] for r in results][::-1]
    hi = [r["hi"] for r in results][::-1]
    colors = [MUTED if r["label"].startswith("Placebo") else BLUE for r in results][::-1]
    fig = go.Figure(go.Scatter(
        x=eff, y=labels, mode="markers", marker=dict(color=colors, size=9),
        error_x=dict(type="data", symmetric=False, array=np.subtract(hi, eff), arrayminus=np.subtract(eff, lo),
                     color=MUTED, thickness=1.5, width=0),
        hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))  # fmt: skip
    fig.add_vline(x=0, line=dict(color=GRID, width=1.5))
    _style(fig, "Change in speed inside the zone, every specification (95% intervals)", height=60 + 42 * len(labels),
           hovermode="closest", showlegend=False, margin=dict(l=20, r=20, t=60, b=40))  # fmt: skip
    fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(automargin=True)
    return fig


def fig_relations(rel: pd.DataFrame) -> go.Figure:
    m = rel.groupby(["month", "relation"])["trips"].sum().reset_index()
    base = m[m["month"].dt.year == 2024].groupby("relation")["trips"].mean()
    fig = go.Figure()
    for r, c in zip(RELATIONS, (ORANGE, VIOLET, YELLOW_SAFE, BLUE), strict=True):
        d = m[m["relation"] == r].sort_values("month")
        if d.empty or r not in base:
            continue
        fig.add_trace(go.Scatter(x=d["month"], y=100 * d["trips"] / base[r], name=r, mode="lines",
                                 line=dict(color=c, width=2), customdata=d["trips"],
                                 hovertemplate="%{x|%b %Y}: %{y:.0f} (%{customdata:,.0f} trips)"))  # fmt: skip
    fig.add_hline(y=100, line=dict(color=GRID, width=1.5))
    _style(fig, "Taxi and ride-hail trips by relation to the zone (2024 average = 100)", height=400,
           y_title="index", legend=dict(orientation="h", y=-0.15, x=0, yanchor="top"),
           margin=dict(l=60, r=20, t=60, b=80))  # fmt: skip
    _toll_line(fig)
    return fig


YELLOW_SAFE = "#b7791f"  # darker yellow: readable as a thin line on a light background


def fig_crz_weekly(daily: pd.DataFrame) -> go.Figure:
    d = daily.copy()
    d["year"] = d["date"].dt.year
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    w = d.groupby(["year", "week"]).agg(entries=("entries", "sum"), days=("date", "nunique")).reset_index()
    w = w[w["days"] == 7]
    fig = go.Figure()
    for y, c in zip(sorted(w["year"].unique()), (BLUE, ORANGE, AQUA), strict=False):
        g = w[w["year"] == y]
        fig.add_trace(go.Scatter(x=g["week"], y=g["entries"], name=str(y), mode="lines", line=dict(color=c, width=2),
                                 hovertemplate=f"{y}, week %{{x}}: %{{y:,.0f}} entries"))  # fmt: skip
    _style(fig, "Vehicles entering the zone per week", height=360, y_title="entries", x_title="week of the year",
           legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(l=70, r=20, t=60, b=80))  # fmt: skip
    return fig


def fig_crz_hours(prof: pd.DataFrame) -> go.Figure:
    p = prof[prof["weekday"]]
    fig = go.Figure()
    for y, c in zip(sorted(p["year"].unique()), (BLUE, ORANGE, AQUA), strict=False):
        g = p[p["year"] == y].sort_values("hour")
        fig.add_trace(go.Bar(x=g["hour"], y=g["per_day"], name=str(y), marker_color=c,
                             hovertemplate=f"{y}, %{{x}}:00: %{{y:,.0f}} per weekday"))  # fmt: skip
    for x in (4.5, 20.5):
        fig.add_vline(x=x, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=12.5, y=1, yref="paper", text="weekday peak toll: 5 a.m. to 9 p.m.", showarrow=False,
                       font=dict(size=11, color=MUTED), yshift=4)  # fmt: skip
    _style(fig, "Entries by hour on an average weekday", height=360, y_title="entries per weekday",
           x_title="hour of the day", barmode="group", bargap=0.15, hovermode="closest",
           legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(l=70, r=20, t=60, b=80))  # fmt: skip
    return fig


def fig_bus(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name, c in (("treated", "Bus segments inside the zone", ORANGE),
                         ("control", "Bus segments outside the zone", BLUE)):  # fmt: skip
        if col in series:
            fig.add_trace(go.Scatter(x=series["month"], y=series[col], name=name, mode="lines+markers",
                                     line=dict(color=c, width=2), marker=dict(size=5),
                                     hovertemplate="%{x|%b %Y}: %{y:.2f} mph"))  # fmt: skip
    _style(fig, "MTA bus speeds (average of routes)", height=360, y_title="mph",
           legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(l=60, r=20, t=60, b=80))  # fmt: skip
    _toll_line(fig)
    return fig


def fig_official(off: pd.DataFrame, mine: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for area, c in zip(sorted(off["area"].unique()), (ORANGE, BLUE, AQUA, VIOLET), strict=False):
        d = off[off["area"] == area].sort_values("month")
        d = d[d["month"] >= mine["month"].min()]
        fig.add_trace(go.Scatter(x=d["month"], y=d["speed"], name=f"MTA: {area}", mode="lines",
                                 line=dict(color=c, width=2), hovertemplate="%{x|%b %Y}: %{y:.2f} mph"))  # fmt: skip
    m = mine[mine["region"] == INSIDE].sort_values("month")
    fig.add_trace(go.Scatter(x=m["month"], y=m["mph"], name="This project: within the zone", mode="lines",
                             line=dict(color=ORANGE, width=2, dash="dot"), hovertemplate="%{x|%b %Y}: %{y:.2f} mph"))  # fmt: skip
    _style(fig, "Cross-check with the MTA's own taxi and ride-hail speed estimates", height=380, y_title="mph",
           legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(l=60, r=20, t=60, b=90))  # fmt: skip
    _toll_line(fig)
    return fig


def fig_map(geo: dict, zc: pd.DataFrame) -> go.Figure:
    zc = zc[zc["zone"].isin([f["id"] for f in geo["features"]])].copy()
    d3 = Z.for_d3(geo)
    metrics = [
        ("speed_change", "Speed change", "Change in average speed of trips from the zone to the same area,<br>"
         "toll months vs. the same months a year earlier", True, 0.15),
        ("pickups_change", "Pickups change", "Change in taxi and ride-hail pickups, toll months vs. a year earlier",
         True, 0.25),
        ("fee_share", "Charged the fee", "Share of trips starting and ending in the zone that paid the per-trip fee",
         False, None),
    ]  # fmt: skip
    custom = np.column_stack([zc["zone_name"], zc["borough"], zc["speed_change"].fillna(np.nan),
                              zc["pickups_change"].fillna(np.nan), zc["fee_share"].fillna(np.nan),
                              np.where(zc["in_zone"], "inside the toll zone", "outside")])  # fmt: skip
    hover = (
        "<b>%{customdata[0]}</b> (%{customdata[1]}, %{customdata[5]})<br>speed %{customdata[2]:+.1%} · "
        "pickups %{customdata[3]:+.1%}<br>fee charged on %{customdata[4]:.0%} of same-zone trips<extra></extra>"
    )
    diverging = [[0, ORANGE], [0.5, "#f0efea"], [1, BLUE]]
    seq = [[0, "#f0efea"], [1, "#104281"]]

    def z_of(col):
        return [None if pd.isna(v) else float(v) for v in zc[col]]

    first = metrics[0]
    fig = go.Figure(go.Choropleth(
        geojson=d3, locations=zc["zone"], featureidkey="id", z=z_of(first[0]), customdata=custom,
        colorscale=diverging, zmid=0, zmin=-first[4], zmax=first[4], hovertemplate=hover,
        marker=dict(line=dict(color=np.where(zc["in_zone"], "#0b0b0b", "#ffffff"), width=np.where(zc["in_zone"], 1.2, 0.5))),
        colorbar=dict(title=dict(text=first[1]), tickformat="+.0%", thickness=12, len=0.75)))  # fmt: skip
    buttons = []
    for col, label, _, div, rng in metrics:
        args = {"z": [z_of(col)], "colorbar.title.text": label,
                "colorbar.tickformat": "+.0%" if div else ".0%",
                "colorscale": [diverging if div else seq], "zmid": [0 if div else None],
                "zmin": [-rng if div else 0], "zmax": [rng if div else 1]}  # fmt: skip
        buttons.append(dict(label=label, method="restyle", args=[args]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="right", type="buttons", x=0, y=1.08,
                                        xanchor="left", yanchor="bottom", bgcolor="#fff", bordercolor=GRID,
                                        font=dict(size=12))])  # fmt: skip
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor=SURFACE)
    _style(fig, "", height=640, hovermode="closest", margin=dict(l=0, r=0, t=60, b=0))
    return fig


ZONE_JS = """
(() => {
  const S = %SERIES%, C = %CONTROL%, map = document.getElementById("zonemap");
  if (!map || !window.Plotly) return;
  const draw = (z) => {
    const s = S[z]; if (!s) { document.getElementById("zone-pick").textContent = "No monthly series for this zone (too few trips)."; return; }
    document.getElementById("zone-pick").innerHTML = "<b>" + s.name + "</b> (" + s.region + ")";
    const t = [
      {x: s.m, y: s.mph, name: "Speed, trips within the area (mph)", mode: "lines+markers", line: {color: "#eb6834", width: 2},
       marker: {size: 4}, yaxis: "y", hovertemplate: "%{x|%b %Y}: %{y:.2f} mph<extra></extra>"},
      {x: C.m, y: C.mph, name: C.name + " (mph)", mode: "lines", line: {color: "#898781", width: 1.5, dash: "dot"},
       yaxis: "y", hovertemplate: "%{x|%b %Y}: %{y:.2f} mph<extra></extra>"},
      {x: s.m, y: s.pickups, name: "Pickups", mode: "lines", line: {color: "#2a78d6", width: 1.5}, yaxis: "y2",
       hovertemplate: "%{x|%b %Y}: %{y:,.0f} pickups<extra></extra>"}];
    Plotly.react("zone-detail", t, {height: 330, margin: {l: 50, r: 60, t: 10, b: 70}, paper_bgcolor: "#fcfcfb",
      plot_bgcolor: "#fcfcfb", font: {family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: "#52514e"},
      yaxis: {title: {text: "mph"}, gridcolor: "#e1e0d9"}, yaxis2: {title: {text: "pickups"}, overlaying: "y", side: "right",
      showgrid: false, tickformat: ",.0f"}, legend: {orientation: "h", y: -0.2}, hovermode: "x unified",
      shapes: [{type: "line", x0: "%TOLL%", x1: "%TOLL%", yref: "paper", y0: 0, y1: 1, line: {color: "#898781", dash: "dot", width: 1}}]},
      {displaylogo: false, responsive: true});
  };
  map.on("plotly_click", (e) => { if (e.points && e.points.length) draw(e.points[0].location); });
  draw(%DEFAULT%);
})();
"""


def zone_series(panel: pd.DataFrame, zt: pd.DataFrame) -> tuple[dict, dict]:
    p = panel.sort_values("month")
    out = {}
    for z, g in p.groupby("zone"):
        out[int(z)] = {"name": str(g["zone_name"].iloc[0]), "region": str(g["region"].iloc[0]),
                       "m": g["month"].dt.strftime("%Y-%m-%d").tolist(),
                       "mph": [None if pd.isna(v) else round(float(v), 2) for v in g["mph"]],
                       "pickups": [None if pd.isna(v) else int(v) for v in g["pickups"]]}  # fmt: skip
    c = p[p["region"] == UPPER]
    c = c.assign(w=c["mph"] * c["speed_trips"]).groupby("month").agg(w=("w", "sum"), n=("speed_trips", "sum"))
    ctrl = {"name": "Average of Manhattan above 60th St", "m": [d.strftime("%Y-%m-%d") for d in c.index],
            "mph": [round(float(v), 2) for v in c["w"] / c["n"]]}  # fmt: skip
    return out, ctrl


# --------------------------------------------------------------------------- page
def build_site(cfg: Config, res: dict) -> dict:
    site = cfg.site_dir
    site.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    months = res["months"]
    first, last = months[0], months[-1]
    results, main = res["results"], res["results"][0]
    placebo = next((r for r in results if r["label"].startswith("Placebo")), None)
    trips_did, bus, crz = res["trips_did"], res["bus"], res["crz"]
    pt = res["pretrend_p"]
    rel_yoy = res["relation_yoy"]
    zt = res["zones"]
    n_inside = int(zt["in_zone"].sum())
    load = res["load_log"]
    rides = int(load["rows_kept"].sum()) if len(load) else 0
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: generated for automated tests; these numbers are not real</div>" if cfg.fixture else ""  # fmt: skip

    cards = [
        _card("Traffic speed inside the zone", _pp(main["effect"]),
              f"vs. Manhattan above 60th St, weekday toll hours · 95% CI {_pp(main['lo'])} to {_pp(main['hi'])}"),
        _card("Taxi and ride-hail pickups in the zone", _pp(trips_did.get("effect")),
              f"vs. Manhattan above 60th St · 95% CI {_pp(trips_did.get('lo'))} to {_pp(trips_did.get('hi'))}"),
    ]  # fmt: skip
    if bus:
        cards.append(_card("Bus speeds inside the zone", _pp(bus["effect"]),
                           f"vs. bus segments outside it · 95% CI {_pp(bus['lo'])} to {_pp(bus['hi'])}"))  # fmt: skip
    if crz and crz.get("yoy"):
        y = crz["yoy"]
        cards.append(_card(f"Vehicle entries, {y['year']} vs. {y['prior']}", _pp(y["change"]),
                           f"Jan 5 to {y['through']} · {_int(y['per_weekday'])} per weekday"))  # fmt: skip
    elif crz:
        cards.append(_card("Vehicles entering per weekday", _int(crz["per_weekday"]), f"since {crz['first_day']}"))
    cards.append(_card("Placebo test (fake toll, Jan 2024)", _pp(placebo["effect"]) if placebo else "–",
                       "should be about zero if the method is sound"))  # fmt: skip
    cards.append(_card("Rides analyzed", f"{rides / 1e6:,.0f}M" if rides >= 1e6 else _int(rides),
                       f"TLC trip records, {first:%b %Y} to {last:%b %Y}"))  # fmt: skip

    did_rows = [[html.escape(r["label"]), _ci(r), f"{r['p']:.3f}" if r["p"] >= 0.001 else "< 0.001",
                 f"{r['zones_inside']} / {r['zones_control']}", _int(r["obs"])] for r in results]  # fmt: skip
    did_rows.append([html.escape(trips_did["label"]), _ci(trips_did), f"{trips_did['p']:.3f}" if trips_did["p"] >= 0.001 else "< 0.001",
                     f"{trips_did['zones_inside']} / {trips_did['zones_control']}", _int(trips_did["obs"])])  # fmt: skip
    ptxt = (f"Before the toll, the inside zones' speeds drifted {_pp(pt['slope_per_year'], 2)} a year relative to the "
            f"control zones (p = {pt['p']:.2f}), {'consistent with' if pt['p'] >= 0.05 else 'a warning sign for'} "
            "the parallel-trends assumption.") if pt and not pd.isna(pt.get("p")) else ""  # fmt: skip

    rel = res["relations"]
    rel_rows = []
    for r in RELATIONS:
        row = [html.escape(r), _pp(rel_yoy.get(r))]
        for s in ("yellow", "fhvhv"):
            d = rel[(rel["relation"] == r) & (rel["service"] == s)].groupby("month", as_index=False)["trips"].sum()
            row.append(_pp(yoy(d, "trips", months)) if len(d) else "–")
        rel_rows.append(row)
    fc = res["fee_check"]
    fee_rows = [[html.escape(r), *[f"{fc.loc[(fc.relation == r) & (fc.service == s), 'share_charged'].iloc[0]:.1%}"
                                   if ((fc.relation == r) & (fc.service == s)).any() else "–"
                                   for s in ("yellow", "green", "fhvhv")]] for r in RELATIONS]  # fmt: skip

    crz_html = "<p class='muted'>The MTA vehicle-entries dataset could not be read on this run.</p>"
    if crz:
        cls_rows = [[html.escape(str(r["name"])), _int(r["current"]), _pp(r.get("change")), f"{r['share']:.0%}"]
                    for r in crz.get("by_class", pd.DataFrame()).to_dict(orient="records")]  # fmt: skip
        ent_rows = [[html.escape(str(r["name"])), _int(r["current"]), _pp(r.get("change"))]
                    for r in crz.get("by_entry", pd.DataFrame()).to_dict(orient="records")]  # fmt: skip
        y = crz.get("yoy")
        cmp = (f"From January 5 to {y['through']}, {_int(y['per_weekday'])} vehicles entered the zone on an average "
               f"weekday in {y['year']}, {_pp(y['change'])} vs. the same days of {y['prior']}." if y else "")  # fmt: skip
        share = crz.get("period_share", {})
        peak = next((v for k, v in share.items() if "peak" in str(k).lower()), None)
        crz_html = f"""<p>The MTA counts every vehicle entering the zone, by entry point, vehicle class and 10-minute
interval, from the day tolling started ({crz["first_day"]} to {crz["last_day"]}). {cmp}
{f"{peak:.0%} of entries happen during peak toll hours." if peak is not None else ""} There is no comparable count
from before the toll, so this shows how traffic has evolved since, not the toll's initial effect.</p>
<div class="grid2">{_div(fig_crz_weekly(crz["daily"]))}{_div(fig_crz_hours(crz["hour_profile"])) if "hour_profile" in crz else ""}</div>
<div class="grid2"><div><h3>By vehicle class{f", {y['year']} vs. {y['prior']}" if y else ""}</h3>
{_table(["Class", "Entries", "Change", "Share"], cls_rows) if cls_rows else ""}</div>
<div><h3>By entry point</h3>{_table(["Entry point", "Entries", "Change"], ent_rows) if ent_rows else ""}</div></div>"""

    bus_html = (
        "<p class='muted'>The MTA bus-speed dataset could not be read on this run, or had no pre-toll months.</p>"
    )
    if bus:
        bus_html = f"""<p>A second, independent source: MTA bus speeds on segments inside vs. outside the zone
({bus["routes"]} routes, {bus["units"]} route segments). The same difference-in-differences gives
<b>{_pp(bus["effect"])}</b> (95% CI {_pp(bus["lo"])} to {_pp(bus["hi"])}) for buses inside the zone.</p>
{_div(fig_bus(bus["series"]))}"""
    off = res.get("taxi_official")
    official_html = (_div(fig_official(off, res["speeds"]["weekday_peak"])) if off is not None and len(off) else
                     "<p class='muted'>The MTA's taxi speed estimates could not be read on this run.</p>")  # fmt: skip

    info = res["cbd_info"]
    agree = info.get("agreement_official_fees")
    zone_def = (f"Zones inside the toll zone: <b>{n_inside}</b> TLC taxi zones, from the {html.escape(info['used'])}. "
                + (f"The MTA's list and the zones where same-zone trips paid the fee agree on {agree:.0%} of zones."
                   if agree is not None else ""))  # fmt: skip
    series, ctrl = zone_series(res["panels"]["all"], zt)
    default = next((z for z in (161, 230, 162) if z in series), next(iter(series), 0))
    zone_js = (ZONE_JS.replace("%SERIES%", json.dumps(series)).replace("%CONTROL%", json.dumps(ctrl))
               .replace("%TOLL%", f"{POLICY_MONTH}-01").replace("%DEFAULT%", str(default)))  # fmt: skip
    notes = res["mta_notes"]
    note_rows = [[k, html.escape(v)] for k, v in notes.items()]
    proc = res["processed_now"]
    load_line = (f"This run downloaded and condensed {len(proc)} new monthly file(s)." if proc else
                 "No new TLC month was published since the last run; everything else was refreshed.")  # fmt: skip
    files = len(load)
    gb = load["bytes"].sum() / 1e9 if len(load) else 0

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NYC Congestion Pricing Impact</title>
<meta name="description" content="Did New York's congestion toll speed up traffic? Difference-in-differences on TLC taxi and ride-hail trip records, MTA vehicle entries and bus speeds.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚦</text></svg>">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>{CSS}{EXTRA_CSS}</style></head>
<body>{banner}
<header><div class="wrap">
  <h1>Did congestion pricing speed up Manhattan?</h1>
  <p>On January 5, 2025, New York started charging vehicles to enter Manhattan south of 60th Street. This page measures
  what changed, using {f"{rides / 1e6:,.0f} million" if rides >= 1e6 else f"{rides:,}"} taxi and ride-hail trips ({first:%b %Y} to {last:%b %Y}), MTA counts of
  vehicles entering the zone, and MTA bus speeds, and separates the toll's effect from seasons and citywide trends.</p>
  <a class="btn" href="#map">Map of every zone</a><a class="btn" href="sql.html">Query the data (SQL in your browser)</a>
  <a class="btn" href="{cfg.repo_url}">Code on GitHub</a>
</div></header>
<main>
<div class="cards">{"".join(cards)}</div>

<section><h2>Did traffic get faster inside the zone?</h2>
<p>Each taxi and ride-hail trip has a start, an end, a distance and a duration. Averaging the speed of trips that stay
inside one area gives a monthly speed for every taxi zone. Speeds inside the toll zone are compared with zones in
Manhattan above 60th Street, which face the same weather, seasons and citywide trends but no toll
(difference-in-differences with zone and month fixed effects, standard errors clustered by zone).</p>
<div class="grid2">{_div(fig_speeds(res["speeds"]["weekday_peak"]))}{_div(fig_event(res["event_study"], "Inside vs. control, month by month (relative to Dec 2024)", "difference in speed"))}</div>
<p class="callout"><b>Result:</b> during weekday toll hours, trips inside the zone became <b>{_pp(main["effect"])}</b>
faster than they would have been without the toll (95% CI {_pp(main["lo"])} to {_pp(main["hi"])}), measured against
Manhattan above 60th Street. {ptxt} A fake toll placed in January 2024 gives {_pp(placebo["effect"]) if placebo else "–"},
as it should if the method isn't picking up something else.</p>
{_div(fig_forest(results))}
{_table(["Specification", "Effect", "p-value", "Zones inside / control", "Zone-months"], did_rows)}
<p class="muted">Raw changes, toll months vs. the same months a year earlier: inside the zone {_pp(res["raw_speed_yoy"].get(INSIDE))},
Manhattan above 60th St {_pp(res["raw_speed_yoy"].get(UPPER))}, Brooklyn, Queens and the Bronx {_pp(res["raw_speed_yoy"].get(OUTER))}.
The difference-in-differences nets out what happened everywhere.</p>
</section>

<section><h2>Buses, and a cross-check with the MTA</h2>
{bus_html}
{official_html}
</section>

<section><h2>Vehicles entering the zone</h2>{crz_html}</section>

<section><h2>Taxi and ride-hail trips</h2>
<p>Taxis and app rides pay a per-trip charge ($0.75 for taxis, $1.50 for Uber, Lyft and other app rides) on every trip
that starts, ends or passes through the zone, at all hours. The trip records show it, which also confirms which
taxi zones are inside: trips that start and end in the same zone pay it only if that zone is in the toll zone.</p>
{_div(fig_relations(rel))}
<div class="grid2"><div><h3>Change in trips, toll months vs. a year earlier</h3>
{_table(["Trips", "All services", SERVICE_NAMES["yellow"], SERVICE_NAMES["fhvhv"]], rel_rows)}</div>
<div><h3>Share of trips charged the per-trip fee since January 2025</h3>
{_table(["Trips", "Yellow", "Green", "App rides"], fee_rows)}
<p class="muted">Trips outside the zone are charged only when they pass through it.</p></div></div>
</section>

<section id="map"><h2>Every taxi zone</h2>
<p>Zones inside the toll zone have a dark outline. Switch between the change in speed, the change in pickups and the
share of same-zone trips charged the fee; click a zone to see its monthly speed and pickups below.</p>
{_div(fig_map(res["geo"], res["zone_change"]), div_id="zonemap")}
<p class="zone-pick" id="zone-pick"></p><div id="zone-detail"></div>
<p class="muted">Changes compare the latest 12 toll months with the same months a year earlier. Zones without
enough trips staying in their area are left uncolored.</p>
</section>

<section><h2>How it works</h2><ul class="method">
<li><b>Data:</b> every NYC TLC trip record for yellow taxis, green taxis and high-volume for-hire vehicles (Uber, Lyft
and others), {first:%B %Y} to {last:%B %Y}: {files} monthly files ({gb:,.0f} GB), each downloaded once, condensed
with DuckDB to zone-to-zone monthly totals and deleted. {load_line} MTA datasets from data.ny.gov: vehicle entries,
bus speeds and the MTA's own taxi speed estimates.</li>
<li><b>Speeds:</b> trips of 1 minute to 3 hours, 0.1 to 100 miles and 0.5 to 75 mph; a zone's speed is total miles over
total hours of trips that start in it and end in the same area. Toll periods follow the toll: weekdays 5 a.m. to
9 p.m., weekends 9 a.m. to 9 p.m., overnight otherwise.</li>
<li><b>Causal design:</b> difference-in-differences with zone and month fixed effects and standard errors clustered
by zone; an event study by month; a pre-toll trend test; a placebo toll a year early; an alternative control group;
a version that lets the zone have its own seasonal pattern; and separate estimates by time of day and service.</li>
<li><b>{zone_def}</b></li>
<li><b>Limits:</b> taxi and ride-hail trips are a sample of traffic, not all of it; speeds are door-to-door averages
including stops; the control areas could be affected indirectly (for example, traffic diverting around the zone),
which would bias the estimate; January 2025 includes four days before the toll.</li>
</ul>
<h3>MTA datasets on this run</h3>{_table(["Dataset", "Status"], note_rows)}
</section>
</main>
<footer>Data: NYC Taxi &amp; Limousine Commission trip records · MTA via data.ny.gov · Built {built_at} ·
<a href="{cfg.repo_url}">Source code</a></footer>
<script>{LAYOUT_JS}</script>
<script>{zone_js}</script>
</body></html>"""
    (site / "index.html").write_text(page, encoding="utf-8")
    (site / ".nojekyll").touch()

    summary = {
        "data_from": str(first.date()), "data_through": str(last.date()), "months": len(months), "rides": rides,
        "zones_inside": n_inside, "zone_definition": info["used"], "zone_agreement": agree,
        "speed_effect": main, "robustness": results[1:], "pretrend": pt, "pickups_effect": trips_did,
        "bus_effect": {k: v for k, v in bus.items() if k != "series"} if bus else None,
        "entries": crz.get("yoy") if crz else None, "entries_per_weekday": crz.get("per_weekday") if crz else None,
        "raw_speed_yoy": res["raw_speed_yoy"], "relation_yoy": rel_yoy, "mta": notes, "synthetic": cfg.fixture,
        "built_at": built_at, "files": files,
    }  # fmt: skip
    return summary
