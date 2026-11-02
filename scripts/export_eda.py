"""Export presentation reports from saved notebook outputs without running cells.

Install requirements/presentation.txt, then run python -m scripts.export_eda.
The exported figures are the original 2025 analysis, not synthetic smoke data.
"""

import argparse
import base64
from html import escape
import json
from pathlib import Path


NOTEBOOK = "full-stack-adhd-classifier-ml-dl-graph-learnin.ipynb"
SECTIONS = {
    "categorical": {
        "title": "Who is represented in the data?",
        "subtitle": "Demographics, family background and the balance of diagnosis and sex groups.",
        "summary": "Read the category distributions first, then compare the four sex and diagnosis groups. These charts describe the study sample.",
        "panels": [
            ("Category distributions", [46], False),
            ("Sex and diagnosis breakdown", [56], False),
            ("Recorded observations", [47, 57, 58], True),
            ("Frequency and cross-tabulation tables", [42, 50], True),
            ("Feature descriptions and analysis notes", [26], True),
        ],
    },
    "quantitative": {
        "title": "Which questionnaire features stand out?",
        "subtitle": "Missing values, feature distributions and correlations with the two targets.",
        "summary": "Compare the feature distributions across groups, then inspect the correlation table and heatmap. Correlations here are exploratory associations within the competition sample.",
        "panels": [
            ("ADHD vs. non-ADHD, split by sex", [83], False),
            ("Female vs. male, split by diagnosis", [85], False),
            ("Feature correlations with each target", [101], False),
            ("Correlation between questionnaire features", [112], False),
            ("Recorded observations", [86, 87, 108, 113, 114], True),
            ("Missing values and descriptive statistics", [64, 70, 73], True),
            ("How to read the correlations", [90, 91], True),
        ],
    },
    "connectome": {
        "title": "How does brain connectivity differ across groups?",
        "subtitle": "Connectivity difference maps, network communities and exploratory graph features.",
        "summary": "Start with the four group difference maps. The interactive networks retain the top 5% of edges by absolute difference; hover over nodes and edges to inspect them.",
        "panels": [
            ("Group difference heatmaps", [175, 176], False),
            ("Interactive connectivity networks", [180, 181, 182, 183], False),
            ("Community structure", [186, 191, 192], False),
            ("Connectivity within communities", [200, 204], False),
            ("Statistical tests on graph features", [214, 216], True),
            ("Recorded interpretations", [184, 187, 193, 201, 205, 215, 217], True),
            ("Background and terminology", [163, 178], True),
        ],
    },
}

STYLE = """
:root{--ink:#152e43;--muted:#526777;--accent:#176753;--line:#dae4e8;--paper:#fff;--bg:#f3f6f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
header,main,footer{max-width:1180px;margin:auto;padding:24px}header{padding-top:44px}nav{display:flex;flex-wrap:wrap;gap:10px}
a{color:var(--accent);text-underline-offset:3px}nav a,.button{display:inline-block;padding:8px 14px;border:1px solid var(--line);border-radius:8px;background:white;text-decoration:none}
nav a[aria-current=page]{background:var(--ink);color:white}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;max-width:950px;margin:24px 0 12px}
h2{font-size:25px;line-height:1.3;margin:0 0 18px}h3{font-size:20px}p{max-width:950px}.eyebrow{letter-spacing:.1em;font-size:12px;font-weight:700;text-transform:uppercase;color:var(--accent)}
.subtitle{font-size:20px;color:var(--muted)}.note{background:#e6f1ed;border-left:4px solid var(--accent);padding:16px 20px;border-radius:4px}
.panel,details{background:var(--paper);padding:26px;border:1px solid var(--line);border-radius:12px;margin-bottom:24px;min-width:0}
summary{cursor:pointer;font-size:20px;font-weight:650}details[open] summary{margin-bottom:24px}.output{margin:16px 0;overflow-x:auto}
img{display:block;max-width:100%;height:auto;margin:20px auto}.figure{margin:20px 0}.source{font-size:12px;color:var(--muted)}
table{border-collapse:collapse;font-size:13px;max-width:100%}td,th{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:#edf3f5}
pre{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}blockquote{margin:12px 0;padding-left:20px;border-left:3px solid var(--line)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}.cards .panel{display:flex;flex-direction:column}.cards .button{margin-top:auto;align-self:flex-start}
.plotly-graph-div{max-width:100%}.modebar{opacity:.7}footer{font-size:13px;color:var(--muted)}
@media(max-width:600px){header,main,footer{padding:18px}.panel,details{padding:16px}.subtitle{font-size:17px}}
@media print{body{background:white}nav,.modebar{display:none!important}.panel,details{border:0;padding:0}img{max-height:250mm}h2,h3{break-after:avoid}.figure{break-inside:avoid}}
"""


def as_text(value):
    return "".join(value) if isinstance(value, list) else str(value)


def plotly_arguments(source):
    """Read the JSON arguments in a saved Plotly.newPlot call without eval()."""
    start = source.find("Plotly.newPlot(")
    if start < 0:
        return None
    remaining = source[start + len("Plotly.newPlot("):].lstrip()
    arguments = []
    decoder = json.JSONDecoder()
    for index in range(4):
        value, end = decoder.raw_decode(remaining)
        arguments.append(value)
        remaining = remaining[end:].lstrip()
        if index < 3:
            if not remaining.startswith(","):
                raise ValueError("Unrecognized Plotly output argument separator")
            remaining = remaining[1:].lstrip()
    return arguments


def render_cell(cell, cell_index, assets, markdown):
    from plotly import io as pio

    if cell["cell_type"] == "markdown":
        return markdown(as_text(cell.get("source", "")), extensions=["tables", "fenced_code"])
    parts = []
    for output_index, output in enumerate(cell.get("outputs", [])):
        data = output.get("data", {})
        if output.get("output_type") == "error":
            raise ValueError(f"Notebook cell {cell_index} has a saved execution error")
        if "text/html" in data:
            source = as_text(data["text/html"])
            arguments = plotly_arguments(source)
            if arguments:
                _, traces, layout, config = arguments
                height = layout.get("height", 650)
                # The saved charts include the original data and layout. Rebuild
                # their wrapper to remove notebook/CDN/MathJax dependencies.
                html = pio.to_html({"data": traces, "layout": layout}, full_html=False,
                                   include_plotlyjs=False, config={**config, "responsive": True},
                                   div_id=f"plot-{cell_index}-{output_index}",
                                   default_height=f"{height}px")
                parts.append(f'<div class="figure">{html}</div>')
            else:
                parts.append(f'<div class="output">{source}</div>')
        elif "image/png" in data:
            filename = f"cell-{cell_index}-{output_index}.png"
            (assets / filename).write_bytes(base64.b64decode(as_text(data["image/png"])))
            parts.append(f'<img src="assets/{filename}" alt="Original notebook figure from cell {cell_index}" loading="lazy">')
        elif "text/markdown" in data:
            parts.append(markdown(as_text(data["text/markdown"]), extensions=["tables"]))
        elif "text/plain" in data:
            parts.append(f'<pre>{escape(as_text(data["text/plain"]))}</pre>')
        elif output.get("output_type") == "stream":
            parts.append(f'<pre>{escape(as_text(output.get("text", "")))}</pre>')
    if parts:
        parts.append(f'<p class="source">Saved notebook cell {cell_index}</p>')
    return "\n".join(parts)


def page(title, content, active="index", plots=False):
    links = [("index", "Overview"), ("categorical", "Demographics"),
             ("quantitative", "Questionnaires"), ("connectome", "Brain connectivity")]
    nav = "".join(f'<a href="{key}.html"' + (' aria-current="page"' if key == active else '') +
                  f'>{label}</a>' for key, label in links)
    script = '<script src="assets/plotly.min.js"></script>' if plots else ''
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} | WiDS 2025 EDA</title><style>{STYLE}</style>{script}</head>
<body><header><div class="eyebrow">WiDS Datathon 2025 · Original analysis</div><nav aria-label="Report navigation">{nav}</nav>
<h1>{escape(title)}</h1></header><main>{content}</main>
<footer>Source: the saved 2025 hackathon notebook. Reports export recorded outputs; no analysis or training cells were re-executed.
<br>Exploratory group comparisons describe this competition sample. The original notebook contains the complete code and historical notes.</footer></body></html>'''
    return "\n".join(line.rstrip() for line in document.splitlines()) + "\n"


def export(notebook_path, output_dir):
    from markdown import markdown
    from plotly.offline import get_plotlyjs

    notebook = json.loads(notebook_path.read_text())
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    manifest = {"source_notebook": notebook_path.name, "source": "saved notebook outputs (2025)", "reports": {}}
    for key, section in SECTIONS.items():
        body = [f'<p class="subtitle">{escape(section["subtitle"])}</p>',
                f'<div class="note">{escape(section["summary"])}</div>']
        exported_cells = []
        for title, cells, collapsed in section["panels"]:
            rendered = "\n".join(render_cell(notebook["cells"][index], index, assets, markdown) for index in cells)
            exported_cells.extend(cells)
            if collapsed:
                body.append(f'<details><summary>{escape(title)}</summary>{rendered}</details>')
            else:
                body.append(f'<section class="panel"><h2>{escape(title)}</h2>{rendered}</section>')
        (output_dir / f"{key}.html").write_text(page(section["title"], "\n".join(body), key, plots=True), encoding="utf-8")
        manifest["reports"][key] = {"notebook_cells": exported_cells}
    cards = "".join(f'<article class="panel"><div class="eyebrow">{key.title()}</div><h2>{escape(section["title"])}</h2>'
                    f'<p>{escape(section["subtitle"])}</p><a class="button" href="{key}.html">Open report →</a></article>'
                    for key, section in SECTIONS.items())
    overview = '<p class="subtitle">Three presentation reports from the original hackathon: explore the study sample, questionnaire features and functional brain connectivity.</p>'
    overview += '<div class="note"><strong>Ready to present.</strong> These reports contain the real saved notebook charts and tables. No dataset, Python installation or GPU is needed to view them. Interactive charts work offline when the assets folder stays beside the HTML files.</div>'
    overview += f'<div class="cards">{cards}</div><section class="panel"><h2>How to use these reports</h2><ol><li>Choose a report above.</li><li>Scroll through its charts and tables; hover or zoom in interactive figures.</li><li>Expand the recorded observations and detailed tables when discussing the results.</li></ol><p>These are archived outputs from the 2025 notebook, not results from a new run of the refactored scripts. Model training and evaluation are documented separately in the repository.</p></section>'
    (output_dir / "index.html").write_text(page("Exploratory data analysis", overview), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported presentation reports to {output_dir / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path(NOTEBOOK))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/eda"))
    args = parser.parse_args()
    export(args.notebook, args.output_dir)


if __name__ == "__main__":
    main()
