"""Small disk-backed replacement for notebook display() and Figure.show()."""

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class Report:
    def __init__(self, directory, title):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parts = [f"<h1>{escape(title)}</h1>"]
        self.counter = 0

    def _name(self, suffix):
        self.counter += 1
        return self.directory / f"{self.counter:03d}{suffix}"

    def display(self, value):
        if isinstance(value, pd.Series):
            value = value.to_frame()
        if isinstance(value, pd.DataFrame):
            path = self._name("_table.csv")
            value.to_csv(path, index=True)
            self.parts.append(f'<p><a href="{path.name}">Full table (CSV)</a></p>')
            self.parts.append(value.head(100).to_html(escape=True))
        else:
            self.parts.append(f"<pre>{escape(str(value))}</pre>")

    def figure(self, figure=None):
        figure = plt.gcf() if figure is None else figure
        if hasattr(figure, "write_html"):
            path = self._name("_plot.html")
            figure.write_html(path, include_plotlyjs=True, full_html=True)
            self.parts.append(f'<p><a href="{path.name}">Interactive plot</a></p>'
                              f'<iframe src="{path.name}" loading="lazy"></iframe>')
        else:
            path = self._name("_plot.png")
            figure.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(figure)
            self.parts.append(f'<img src="{path.name}" alt="EDA figure">')

    def finish(self):
        destination = self.directory / "index.html"
        destination.write_text(
            '<!doctype html><html><head><meta charset="utf-8"><style>'
            'body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:16px;}'
            'img,iframe{width:100%;border:0;}iframe{height:750px;}'
            'table{border-collapse:collapse;font-size:12px;}td,th{padding:5px;}'
            'pre{white-space:pre-wrap;}table{display:block;overflow-x:auto;}'
            '</style></head><body>' + "\n".join(self.parts) + '</body></html>',
            encoding="utf-8")
        print(f"EDA report: {destination}")
        return destination
