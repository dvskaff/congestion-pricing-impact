"""Rewrites the results block in README.md after each run."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _pv(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _p(v, d=1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _ci(r: dict) -> str:
    return f"{_p(r['effect'])} ({_p(r['lo'])} to {_p(r['hi'])})"


def results_block(s: dict) -> str:
    sp, rob, pk, bus = s["speed"], s["robustness"], s["pickups"], s.get("bus")
    lines = [
        f"_TLC trip records {pd.Timestamp(s['data_from']):%b %Y} to {pd.Timestamp(s['data_through']):%b %Y} · "
        f"{s['rides'] / 1e6:,.0f}M taxi and ride-hail trips · {s['zones_inside']} taxi zones inside the toll zone "
        f"({s['zone_definition']}) · last run {s['built_at']}_",
        "",
        "**Speed of taxi and app-ride trips inside the zone**, weekday toll hours, vs. Manhattan above 60th St:",
        "",
        "| Estimate | Effect (95% CI) |",
        "|---|---|",
    ]
    for key in ("first", "second", "trend", "placebo", "pooled"):
        r = sp.get(key)
        if r:
            bold = "**" if key in ("first", "trend") else ""
            lines.append(f"| {bold}{r['label']}{bold} | {bold}{_ci(r)}{bold} |")
    if rob:
        first = sp.get("first", {}).get("label", "").split(": ")[-1]
        lines += ["", f"**Robustness checks**, speed inside the zone, {first}:", "", "| Specification | Effect (95% CI) |",
                  "|---|---|"]  # fmt: skip
        lines += [f"| {r['label'].rsplit(', ', 1)[0]} | {_ci(r)} |" for r in rob]
    other = [("Taxi and app-ride pickups inside the zone", pk.get(k)) for k in ("first", "second")]
    other += [
        ("MTA bus speeds, route segments inside vs. outside the zone", (bus or {}).get(k)) for k in ("first", "trend")
    ]
    other = [(name, r) for name, r in other if r]
    if other:
        lines += ["", "**Other outcomes**, same design:", "", "| Outcome | Effect (95% CI) |", "|---|---|"]
        lines += [f"| {name}: {r['label'].split(': ', 1)[1]} | {_ci(r)} |" for name, r in other]
    lines.append("")
    pt = s.get("pretrend") or {}
    if pt.get("p") is not None and not pd.isna(pt.get("p")):
        drift = pt["p"] < 0.05
        tail = ("so the year-over-year and trend-adjusted estimates bracket the toll's effect"
                if drift else "consistent with the two areas moving in parallel")  # fmt: skip
        lines.append(f"**Before the toll,** inside-zone speeds drifted {_p(pt['slope_per_year'])} a year relative to "
                     f"the control zones ({_pv(pt['p'])}), {tail}; see the dashboard for the month-by-month picture.")  # fmt: skip
        lines.append("")
    rel = s.get("relation_yoy", {})
    if rel:
        lines.append("**Trips, latest 12 toll months vs. a year earlier:** "
                     + " · ".join(f"{k.lower()} {_p(v)}" for k, v in rel.items()))  # fmt: skip
        lines.append("")
    e = s.get("entries")
    if e:
        lines.append(f"**Vehicles entering the zone (MTA):** {e['per_weekday']:,.0f} per weekday in {e['year']}, "
                     f"{_p(e['change'])} vs. the same days of {e['prior']} (Jan 5 to {e['through']}).")  # fmt: skip
    elif s.get("entries_per_weekday"):
        lines.append(f"**Vehicles entering the zone (MTA):** {s['entries_per_weekday']:,.0f} per weekday.")
    return "\n".join(lines)


def update_readme(readme: Path, summary: dict) -> None:
    text = readme.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)"
    readme.write_text(re.sub(pattern, lambda m: m.group(1) + results_block(summary) + m.group(2), text, flags=re.S))
