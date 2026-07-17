#!/usr/bin/env python3
"""Build a readable .odt report from WOLF result JSON files.

Writes the OpenDocument file directly (an .odt is a zip of ODF XML), so no
LibreOffice or odfpy is required.

Usage:
  python3 make_report.py results/LYFT.json results/PLTR.json results/MSFT.json \
      --out WOLF_report.odt [--sample]
"""

import argparse
import json
import zipfile
from xml.sax.saxutils import escape

STYLES = """
<style:style style:name="Title" style:family="paragraph">
 <style:paragraph-properties fo:margin-bottom="0.15in" fo:border-bottom="2.5pt solid #2c3e70" fo:padding-bottom="0.06in"/>
 <style:text-properties fo:font-size="22pt" fo:font-weight="bold" fo:color="#1a1a2e"/>
</style:style>
<style:style style:name="H2" style:family="paragraph">
 <style:paragraph-properties fo:margin-top="0.25in" fo:margin-bottom="0.08in"/>
 <style:text-properties fo:font-size="16pt" fo:font-weight="bold" fo:color="#2c3e70"/>
</style:style>
<style:style style:name="H3" style:family="paragraph">
 <style:paragraph-properties fo:margin-top="0.15in" fo:margin-bottom="0.05in"/>
 <style:text-properties fo:font-size="12pt" fo:font-weight="bold" fo:color="#444444"/>
</style:style>
<style:style style:name="P" style:family="paragraph">
 <style:paragraph-properties fo:margin-bottom="0.08in"/>
 <style:text-properties fo:font-size="10.5pt"/>
</style:style>
<style:style style:name="Note" style:family="paragraph">
 <style:paragraph-properties fo:margin-bottom="0.08in"/>
 <style:text-properties fo:font-size="9pt" fo:color="#555555" fo:font-style="italic"/>
</style:style>
<style:style style:name="Banner" style:family="paragraph">
 <style:paragraph-properties fo:background-color="#b30000" fo:text-align="center" fo:padding="0.08in" fo:margin-bottom="0.15in"/>
 <style:text-properties fo:font-size="12pt" fo:font-weight="bold" fo:color="#ffffff"/>
</style:style>
<style:style style:name="THText" style:family="paragraph">
 <style:text-properties fo:font-size="9.5pt" fo:font-weight="bold" fo:color="#ffffff"/>
</style:style>
<style:style style:name="TDText" style:family="paragraph">
 <style:text-properties fo:font-size="9.5pt"/>
</style:style>
<style:style style:name="TDTextWin" style:family="paragraph">
 <style:text-properties fo:font-size="9.5pt" fo:font-weight="bold" fo:color="#0a7a2f"/>
</style:style>
<style:style style:name="TH" style:family="table-cell">
 <style:table-cell-properties fo:background-color="#2c3e70" fo:border="0.5pt solid #2c3e70" fo:padding="0.04in"/>
</style:style>
<style:style style:name="TD" style:family="table-cell">
 <style:table-cell-properties fo:border="0.5pt solid #c5cbe0" fo:padding="0.04in"/>
</style:style>
<style:style style:name="Tbl" style:family="table">
 <style:table-properties style:width="6.7in" fo:margin-bottom="0.12in"/>
</style:style>
"""

CONTENT_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 office:version="1.2">
 <office:automatic-styles>{styles}</office:automatic-styles>
 <office:body><office:text>{body}</office:text></office:body>
</office:document-content>"""

MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
 <manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""


def p(text, style="P"):
    return f'<text:p text:style-name="{style}">{text}</text:p>'


def h(text, style):
    return p(escape(text), style)


def table(header, rows, win_col=None):
    out = [f'<table:table table:style-name="Tbl">'
           f'<table:table-column table:number-columns-repeated="{len(header)}"/>']
    out.append("<table:table-row>")
    for cell in header:
        out.append(f'<table:table-cell table:style-name="TH" office:value-type="string">'
                   f'{p(escape(str(cell)), "THText")}</table:table-cell>')
    out.append("</table:table-row>")
    for row in rows:
        out.append("<table:table-row>")
        for i, cell in enumerate(row):
            style = "TDTextWin" if i == win_col else "TDText"
            out.append(f'<table:table-cell table:style-name="TD" office:value-type="string">'
                       f'{p(escape(str(cell)), style)}</table:table-cell>')
        out.append("</table:table-row>")
    out.append("</table:table>")
    return "".join(out)


def fmt_params(params):
    return ", ".join(f"{k} = {v}" for k, v in params.items())


def fmt_if(v):
    if v is None or v == float("inf"):
        return "∞ (no losers)"
    return f"{v:.2f}"


def metric_cells(m):
    return [f"{m['win_pct']:.1f}%", fmt_if(m["impact_factor"]),
            f"{m['max_dd_points']:.2f}", f"{m['total_points']:.2f}",
            f"{m['points_per_trade']:.2f}", m["trades"]]


def section(res):
    body = [h(res["ticker"], "H2")]
    body.append(p(escape(
        f"Data: {res['date_range'][0]} → {res['date_range'][1]} ({res['bars']} trading days) · "
        f"last close {res['last_close']:.2f} · buy & hold over the period: "
        f"{res['buy_hold_points']:.2f} points · "
        f"{res['combos_tested']} strategy/parameter combinations tested"), "Note"))
    best = res["best"]
    if not best:
        body.append(p("No strategy met the minimum-trade filter.", "P"))
        return "".join(body)
    body.append(h(f"Most profitable strategy: {best['strategy']} ({fmt_params(best['params'])})", "H3"))
    body.append(table(
        ["Win%", "IF (profit factor)", "Max DD (pts)", "Total Points", "Points / Trade", "Trades"],
        [metric_cells(best)], win_col=3))
    body.append(h("Runners-up", "H3"))
    body.append(table(
        ["Strategy", "Parameters", "Win%", "IF", "Max DD", "Total Pts", "Pts/Trade", "Trades"],
        [[r["strategy"], fmt_params(r["params"])] + metric_cells(r) for r in res["top"][1:]],
        win_col=5))
    return "".join(body)


def build_body(results, sample):
    body = []
    if sample:
        body.append(p("⚠ SAMPLE REPORT — GENERATED FROM SYNTHETIC DEMONSTRATION DATA, "
                      "NOT REAL MARKET PRICES ⚠", "Banner"))
    body.append(h("WOLF Strategy Optimization Report", "Title"))
    body.append(p(escape(
        "WOLF (Wide Optimization of Long/Flat strategies) grid-searches seven families of daily "
        "technical strategies — SMA cross, EMA cross, RSI mean reversion, Donchian breakout, "
        "MACD, Bollinger mean reversion and momentum — and ranks every parameter combination by "
        "total points captured. All results are long/flat, one unit, close-to-close, and include "
        "no costs unless stated."), "P"))
    for res in results:
        body.append(section(res))
    body.append(h("Definitions", "H2"))
    body.append(table(["Metric", "Meaning"], [
        ["Win%", "Winning trades ÷ total trades"],
        ["IF", "Impact factor (profit factor): gross winning points ÷ gross losing points"],
        ["Max DD", "Deepest peak-to-trough drop of the strategy equity curve, in points"],
        ["Total Points", "Sum of points across all trades (1 point = $1 per share)"],
        ["Points/Trade", "Total points ÷ number of trades"],
    ]))
    body.append(h("Important caveats", "H2"))
    body.append(p(escape(
        "These are the best in-sample parameter sets: the metrics are optimistically biased by "
        "the optimization itself (overfitting). Past performance does not predict future "
        "results. No slippage, commissions or dividends are modeled unless a per-trade cost was "
        "specified. This report is research output, not investment advice."), "Note"))
    return "".join(body)


def write_odt(path, body_xml):
    content = CONTENT_TEMPLATE.format(styles=STYLES, body=body_xml)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"),
                   "application/vnd.oasis.opendocument.text",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/manifest.xml", MANIFEST,
                   compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("content.xml", content, compress_type=zipfile.ZIP_DEFLATED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--out", default="WOLF_report.odt")
    ap.add_argument("--sample", action="store_true",
                    help="stamp the report as synthetic sample data")
    args = ap.parse_args()

    results = []
    for path in args.jsons:
        with open(path) as f:
            results.append(json.load(f))
    write_odt(args.out, build_body(results, args.sample))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
