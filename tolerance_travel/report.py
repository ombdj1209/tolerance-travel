"""Self-contained HTML report generated from results.json."""
from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root{--ink:#10243e;--muted:#566479;--line:#dfe5ee;--bg:#fbfcfe;--hi:#0b5cad}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Source Sans 3","Segoe UI",system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:48px 24px 80px}h1{font-size:2rem;line-height:1.15;margin:0 0 8px}
h2{font-size:1.2rem;margin:44px 0 8px}p{max-width:70ch;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.n{text-align:right}.wrap{overflow-x:auto}td.lo{color:#a33a1f;font-weight:600}
"""


def _t(headers, rows):
    h = "".join(f'<th class="{"n" if i else ""}">{html.escape(x)}</th>' for i, x in enumerate(headers))
    b = "".join("<tr>" + "".join(c if c.startswith("<td") else f'<td class="{"n" if i else ""}">{c}</td>'
                                 for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def write_report(r: dict, path: Path) -> None:
    sysrows = [[s, f'{v["success"]:.1%}', f'{v["constraint_violation"]:.1%}', f'€{1000 * v["cost_eur_per_task"]:.2f}',
                f'{v["latency_p50_s"]:.1f} s', f'{v["tool_calls"]:.1f}'] for s, v in r["by_system"].items()]
    systems = list(r["by_system"])
    mrows = []
    for m in r["mutations"]:
        row = [m]
        for s in systems:
            v = r["by_mutation"].get(f"{m}/{s}")
            if v is None:
                row.append("")
                continue
            cls = ' class="n lo"' if v["success"] < 0.5 else ' class="n"'
            row.append(f'<td{cls}>{v["success"]:.0%}</td>')
        mrows.append(row)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tolerance-travel results</title><style>{CSS}</style></head><body><main>
<h1>Does the agent earn its cost?</h1>
<p>{r["tasks"]["n"]} synthetic trip-planning tasks ({html.escape(str(r["tasks"]["by_kind"]))}), exact oracle answers,
{len(r["mutations"])} tool-spec conditions. Runtime {r["environment"]["runtime_s"]} s.</p>
<h2>Stable API</h2>{_t(["System", "Success", "Constraint violations", "Cost per 1,000 tasks", "p50 latency", "Tool calls"], sysrows)}
<h2>Success under tool-spec changes</h2><p>Each row changes the published spec and what the server accepts.</p>
{_t(["Spec change"] + systems, mrows)}
</main></body></html>"""
    Path(path).write_text(doc)
