"""Rewrites the results block in README.md after each run."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _p(v, d=1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _ci(r: dict) -> str:
    return f"{_p(r['effect'])} ({_p(r['lo'])} to {_p(r['hi'])})"


def results_block(s: dict) -> str:
    main, rob, trips = s["speed_effect"], s["robustness"], s["pickups_effect"]
    lines = [
        f"_TLC trip records {pd.Timestamp(s['data_from']):%b %Y} to {pd.Timestamp(s['data_through']):%b %Y} · "
        f"{s['rides'] / 1e6:,.0f}M taxi and ride-hail trips · {s['zones_inside']} taxi zones inside the toll zone "
        f"({s['zone_definition']}) · last run {s['built_at']}_",
        "",
        "| Estimate (difference-in-differences, 95% CI) | Effect of the toll |",
        "|---|---|",
        f"| **Speed of trips inside the zone**, weekday toll hours, vs. Manhattan above 60th St | **{_ci(main)}** |",
    ]
    for r in rob:
        lines.append(f"| {r['label']} | {_ci(r)} |")
    lines.append(f"| Taxi and ride-hail pickups inside the zone | {_ci(trips)} |")
    if s.get("bus_effect"):
        lines.append(f"| MTA bus speeds, segments inside vs. outside the zone | {_ci(s['bus_effect'])} |")
    lines.append("")
    pt = s.get("pretrend") or {}
    if pt.get("p") is not None and not pd.isna(pt.get("p")):
        lines.append(f"**Parallel trends:** before the toll, inside-zone speeds drifted {_p(pt['slope_per_year'], 2)} "
                     f"a year relative to the control zones (p = {pt['p']:.2f}).")  # fmt: skip
        lines.append("")
    rel = s.get("relation_yoy", {})
    if rel:
        lines.append("**Trips, toll months vs. a year earlier:** "
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
