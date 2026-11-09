"""Query the results in the browser: exports the analysis tables for DuckDB-WASM and writes the SQL page (sql.html).

The tables are written as gzipped CSV (DuckDB in the browser reads these with no extensions; they double as data
downloads) and described in data/tables.json with each column's type and meaning. The page loads them into an
in-browser DuckDB with those types and runs whatever SQL the visitor types.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)
RUNTIME = ["duckdb.bundle.mjs", "duckdb-eh.wasm", "duckdb-browser-eh.worker.js", "duckdb-mvp.wasm",
           "duckdb-browser-mvp.worker.js"]  # fmt: skip
COLUMNS = {
    "month": "First day of the month", "date": "Day", "service": "yellow, green or fhvhv (Uber, Lyft and other app rides)",
    "zone": "TLC taxi zone ID", "zone_name": "Taxi zone name", "borough": "Borough",
    "region": "Analysis area: inside the toll zone, Manhattan above 60th St, Brooklyn/Queens/Bronx, or excluded",
    "in_zone": "Inside the Congestion Relief Zone", "pickups": "Trips starting in the zone",
    "dropoffs": "Trips ending in the zone", "period": "weekday_peak (5 a.m.-9 p.m.), weekend_peak (9 a.m.-9 p.m.) or overnight",
    "relation": "Within the zone, into it, out of it, or outside it", "trips": "Trips",
    "fee_trips": "Trips charged the per-trip congestion fee", "fee_sum": "Per-trip congestion fees charged ($)",
    "speed_trips": "Trips with a plausible speed (used for speeds)", "miles": "Miles of those trips",
    "hours": "Hours of those trips", "mph": "Average speed: miles / hours", "specification": "Which estimate",
    "effect": "Estimated change (0.05 = +5%)", "lo": "95% interval, low", "hi": "95% interval, high",
    "p_value": "p-value", "zones_inside": "Zones inside in the estimate", "zones_control": "Control zones",
    "observations": "Zone-months (or route-months) used", "speed_change": "Change in speed, toll months vs. a year earlier",
    "pickups_change": "Change in pickups, toll months vs. a year earlier",
    "fee_share": "Share of same-zone trips charged the fee since Jan 2025", "mph_after": "Speed in the toll months",
    "vehicle_class": "Vehicle class (MTA)", "entry_point": "Where vehicles entered (MTA detection group)",
    "entries": "Vehicles entering the zone", "route": "Bus route", "speed": "Average bus speed (mph)",
    "n": "Bus trips in the average", "service_zone": "TLC service zone",
}  # fmt: skip


def tables(res: dict) -> dict[str, tuple[pd.DataFrame, str]]:
    """The tables published for the SQL page and as downloads."""
    zc = res["zone_change"]
    out = {
        "zones": (zc[["zone", "zone_name", "borough", "region", "in_zone", "speed_change", "pickups_change",
                      "fee_share", "mph_after"]], "Every taxi zone, whether it is inside the toll zone, and how its "
                  "speed and pickups changed (latest 12 toll months vs. the same months a year earlier)."),
        "zone_month": (res["zone_month"], "Pickups and dropoffs per taxi zone, month and service."),
        "zone_speeds": (res["panels"]["all"][["zone", "month", "region", "mph", "speed_trips", "pickups"]],
                        "The difference-in-differences panel: monthly speed of weekday toll-hour trips from each zone to "
                        "the same area, for zones with enough trips."),
        "trips_by_relation": (res["relations"], "Trips by month, service, toll period and relation to the zone, with "
                              "per-trip fees and the miles and hours used for speeds."),
        "area_speeds": (pd.concat([s.assign(period=p) for p, s in res["speeds"].items()])[["month", "period", "region",
                        "mph", "trips"]], "Average speed of trips within each area by month and toll period."),
        "estimates": (res["estimates"], "The toll's estimated effect: every specification, with 95% intervals."),
        "event_study": (res["event_study"][["month", "effect", "lo", "hi"]],
                        "Inside vs. control speed difference by month, relative to December 2024."),
    }  # fmt: skip
    crz = res.get("crz")
    if crz is not None and res.get("daily_detail") is not None:
        out["vehicle_entries"] = (res["daily_detail"], "Vehicles entering the zone per day by class, entry point and "
                                  "toll period (MTA).")  # fmt: skip
    if res.get("bus_raw") is not None:
        out["bus_speeds"] = (res["bus_raw"], "Monthly MTA bus speed by route and relation to the zone (MTA).")
    return out


EXAMPLES = [
    ("Speeds by area and year",
     "select year(month) as year, region,\n       round(sum(mph * trips) / sum(trips), 2) as mph\nfrom area_speeds\n"
     "where period = 'weekday_peak'\ngroup by all\norder by region, year;"),
    ("The toll's effect, every estimate", "select specification, round(effect * 100, 1) as pct,\n"
     "       round(lo * 100, 1) as low, round(hi * 100, 1) as high, p_value\nfrom estimates;"),
    ("Zones in the toll zone, by change in speed",
     "select zone_name, round(speed_change * 100, 1) as speed_pct,\n       round(pickups_change * 100, 1) as pickups_pct\n"
     "from zones\nwhere in_zone\norder by speed_change desc;"),
    ("Trips into the zone each month", "select month, service, sum(trips) as trips\nfrom trips_by_relation\n"
     "where relation = 'Into the zone'\ngroup by all\norder by month, service;"),
    ("Per-trip fees paid by riders", "select year(month) as year, service, sum(fee_trips) as trips_charged,\n"
     "       round(sum(fee_sum)) as fees_usd\nfrom trips_by_relation\nwhere month >= '2025-01-01'\n"
     "group by all\norder by year, service;"),
    ("Entries by vehicle class", "select date_trunc('month', date) as month, vehicle_class, sum(entries) as entries\n"
     "from vehicle_entries\ngroup by all\norder by month, entries desc;"),
    ("Month-by-month difference", "select month, round(effect * 100, 1) as pct, round(lo * 100, 1) as low,\n"
     "       round(hi * 100, 1) as high\nfrom event_study\norder by month;"),
]  # fmt: skip


def export(tbls: dict, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    meta = []
    for name, (df, desc) in tbls.items():
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.date
        con.register("t", df)
        cols = con.execute("DESCRIBE SELECT * FROM t").fetchall()
        sel = ", ".join(f'round("{c[0]}", 5) AS "{c[0]}"' if c[1] in ("DOUBLE", "FLOAT") else f'"{c[0]}"' for c in cols)
        path = out_dir / f"{name}.csv.gz"
        con.execute(f"COPY (SELECT {sel} FROM t) TO '{path.as_posix()}' (HEADER, COMPRESSION gzip)")
        con.unregister("t")
        meta.append({"name": name, "file": f"data/{path.name}", "rows": len(df), "bytes": path.stat().st_size,
                     "description": desc,
                     "columns": [{"name": c[0], "type": "BIGINT" if c[1] == "HUGEINT" else c[1],
                                  "description": COLUMNS.get(c[0], "")} for c in cols]})  # fmt: skip
    con.close()
    log.info("Browser SQL: %d tables (%.1f MB)", len(meta), sum(m["bytes"] for m in meta) / 1e6)
    return meta


def copy_runtime(vendor: Path, site: Path) -> bool:
    if not all((vendor / f).exists() for f in RUNTIME):
        log.warning("DuckDB-WASM runtime not found in %s; the SQL page will show a notice", vendor)
        return False
    (site / "vendor").mkdir(parents=True, exist_ok=True)
    for f in RUNTIME:
        shutil.copy(vendor / f, site / "vendor" / f)
    return True


def create_sql(t: dict, path: str) -> str:
    cols = ", ".join(f"'{c['name']}': '{c['type']}'" for c in t["columns"])
    return f"create table {t['name']} as select * from read_csv('{path}', header = true, columns = {{{cols}}})"


# --------------------------------------------------------------------------- page
PAGE_CSS = """
.sqlgrid { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:16px; }
@media (max-width: 900px) { .sqlgrid { grid-template-columns:minmax(0,1fr); } }
textarea#sql { width:100%; min-height:190px; font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  padding:10px 12px; border:1px solid #c3c2b7; border-radius:10px; background:#fff; resize:vertical; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:8px 0 10px; }
.run { background:#104281; color:#fff; border:0; border-radius:9px; padding:8px 16px; font-weight:600; cursor:pointer; }
.run:disabled { opacity:.55; cursor:default; }
.ex { border:1px solid #c3c2b7; background:#fff; border-radius:999px; padding:3px 11px; font-size:13px;
  color:#104281; cursor:pointer; }
.ghost { border:1px solid #c3c2b7; background:#fff; border-radius:9px; padding:7px 12px; font-size:13px; cursor:pointer; }
#status { font-size:13px; color:#52514e; } #status.err { color:#c4352b; white-space:pre-wrap; }
.tables details { border-bottom:1px solid #e1e0d9; padding:6px 0; }
.tables summary { color:#0b0b0b; font-weight:600; margin:0; }
.tables summary span { color:#898781; font-weight:400; font-size:12px; }
.tables p { font-size:12.5px; color:#52514e; margin:4px 0; }
.tables ul { padding-left:18px; margin:4px 0; } .tables li { font-size:12.5px; } .tables code { font-size:12px; }
.results { max-height:520px; overflow:auto; border:1px solid #e1e0d9; border-radius:10px; }
.results table { margin:0; } .results th { position:sticky; top:0; background:#fcfcfb; }
"""

PAGE_JS = """
const $ = (id) => document.getElementById(id);
const status = (t, err = false) => { $("status").textContent = t; $("status").className = err ? "err" : ""; };
EXAMPLES.forEach(([label, sql]) => {
  const b = document.createElement("button"); b.className = "ex"; b.textContent = label;
  b.onclick = () => { $("sql").value = sql; run(); }; $("examples").appendChild(b);
});
let conn = null, last = null;
// Arrow hands dates and timestamps back as epoch milliseconds: format them by column type
const kind = (f) => { const t = String(f.type); return t.startsWith("Date") ? "date" : t.startsWith("Timestamp") ? "ts"
  : /^(Int|Uint|Float|Decimal)/.test(t) ? "num" : "other"; };
const raw = (v, k) => v === null || v === undefined ? "" : k === "date" ? new Date(Number(v)).toISOString().slice(0, 10)
  : k === "ts" ? new Date(Number(v)).toISOString().slice(0, 19).replace("T", " ") : typeof v === "bigint" ? v.toString() : String(v);
const fmt = (v, k) => k === "plain" ? raw(v, k) : typeof v === "number" && k !== "date" && k !== "ts"
  ? (Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, {maximumFractionDigits: 4}))
  : typeof v === "bigint" ? v.toLocaleString() : raw(v, k);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const createSql = (t, path) => `create table ${t.name} as select * from read_csv('${path}', header = true, columns = {`
  + t.columns.map((c) => `'${c.name}': '${c.type}'`).join(", ") + "})";
async function init() {
  if (!RUNTIME) throw new Error("The DuckDB runtime is not bundled in this build.");
  const duckdb = await import("./vendor/duckdb.bundle.mjs");
  const base = new URL("./vendor/", location.href).href;
  const bundle = await duckdb.selectBundle({
    mvp: { mainModule: base + "duckdb-mvp.wasm", mainWorker: base + "duckdb-browser-mvp.worker.js" },
    eh: { mainModule: base + "duckdb-eh.wasm", mainWorker: base + "duckdb-browser-eh.worker.js" },
  });
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(bundle.mainWorker));
  await db.instantiate(bundle.mainModule);
  // DECIMAL results (e.g. 1.5, sums of integers) come back from Arrow unscaled unless cast to double
  await db.open({ query: { castDecimalToDouble: true } });
  conn = await db.connect();
  const meta = await (await fetch("data/tables.json")).json();
  for (const t of meta.tables) {
    status(`Loading ${t.name} (${t.rows.toLocaleString()} rows)…`);
    const buf = new Uint8Array(await (await fetch(t.file)).arrayBuffer());
    await db.registerFileBuffer(t.name + ".csv.gz", buf);
    await conn.query(createSql(t, t.name + ".csv.gz"));
  }
  status(`Ready: ${meta.tables.length} tables loaded.`);
  $("run").disabled = false;
}
async function run() {
  if (!conn) return;
  $("run").disabled = true; status("Running…");
  const t0 = performance.now();
  try {
    const res = await conn.query($("sql").value);
    const cols = res.schema.fields.map((f) => f.name);
    // IDs, years, flight numbers and ranks are labels, not quantities: no thousands separators
    const kinds = res.schema.fields.map((f) => /(^|_)(year|id|number|rank|code)$/i.test(f.name) ? "plain" : kind(f));
    const rows = res.toArray();
    last = {cols, kinds, rows};
    const cls = (i) => kinds[i] === "num" || kinds[i] === "plain" ? "" : "l";  // numbers right-aligned
    const head = "<tr>" + cols.map((c, i) => `<th class="${cls(i)}">${esc(c)}</th>`).join("") + "</tr>";
    const body = rows.slice(0, 1000).map((r) => "<tr>" + cols.map((c, i) =>
      `<td class="${cls(i)}">${esc(fmt(r[c], kinds[i]))}</td>`).join("") + "</tr>").join("");
    $("results").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
    status(`${rows.length.toLocaleString()} row${rows.length === 1 ? "" : "s"} in ${(performance.now() - t0).toFixed(0)} ms`);
    $("csv").disabled = rows.length === 0;
  } catch (e) { status(e.message, true); }
  finally { $("run").disabled = false; }
}
$("run").onclick = run;
$("sql").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); } });
$("csv").onclick = () => {
  if (!last) return;
  const q = (s) => /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  const csv = [last.cols.map(q).join(","),
    ...last.rows.map((r) => last.cols.map((c, i) => q(raw(r[c], last.kinds[i]))).join(","))].join("\\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  a.download = "query_results.csv"; a.click();
};
init().then(() => run()).catch((e) => status("Could not start DuckDB here: " + e.message, true));
"""


def write_page(site: Path, meta: list[dict], as_of: str, runtime_ok: bool, repo_url: str, synthetic: bool) -> None:
    from .report_base import CSS

    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "tables.json").write_text(json.dumps({"as_of": as_of, "tables": meta}, indent=1))
    names = {m["name"] for m in meta}
    examples = [e for e in EXAMPLES if any(f"from {n}" in e[1] for n in names)]
    tables_html = "".join(
        f"<details><summary><code>{m['name']}</code> <span>{m['rows']:,} rows</span></summary>"
        f"<p>{html.escape(m['description'])} <a href='{m['file']}' download>Download (CSV, gzipped)</a></p><ul>"
        + "".join(f"<li><code>{c['name']}</code> <span class='muted'>{c['type'].lower()}</span>"
                  f"{' – ' + html.escape(c['description']) if c['description'] else ''}</li>" for c in m["columns"])
        + "</ul></details>"
        for m in meta
    )  # fmt: skip
    total_mb = sum(m["bytes"] for m in meta) / 1e6
    banner = "<div class='synthetic'>SYNTHETIC TEST DATA: built by the test suite</div>" if synthetic else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Query the Congestion Pricing Data</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚦</text></svg>">
<style>{CSS}{PAGE_CSS}</style></head><body>{banner}
<header><div class="wrap"><h1>Query the data</h1>
<p>Write SQL against the tables behind the congestion pricing analysis: speeds, trips and fees by taxi zone and month,
the estimates, and the MTA's vehicle entries. DuckDB runs inside your browser on files exported after the latest run
(data through {html.escape(as_of)}), so nothing is sent to a server. Each table can also be downloaded.</p>
<a class="btn" href="./">Back to the dashboard</a><a class="btn" href="{repo_url}">Code on GitHub</a></div></header>
<main><section><div class="sqlgrid"><div>
<div class="bar" id="examples"></div>
<textarea id="sql" spellcheck="false">{html.escape(examples[0][1])}</textarea>
<div class="bar"><button class="run" id="run" disabled>Run (Ctrl+Enter)</button>
<button class="ghost" id="csv" disabled>Download results as CSV</button><span id="status">Loading DuckDB…</span></div>
<div class="results" id="results"></div>
<p class="muted">The first visit downloads DuckDB and the tables ({total_mb:.1f} MB); your browser caches them
afterwards. Results show up to 1,000 rows; the CSV download has them all.</p>
</div><div class="tables"><h3>Tables</h3>{tables_html}</div></div></section></main>
<footer>Data: NYC Taxi &amp; Limousine Commission trip records · MTA via data.ny.gov · <a href="{repo_url}">Source code</a></footer>
<script type="module">
const EXAMPLES = {json.dumps(examples)};
const RUNTIME = {"true" if runtime_ok else "false"};
{PAGE_JS}
</script></body></html>"""
    (site / "sql.html").write_text(page, encoding="utf-8")
