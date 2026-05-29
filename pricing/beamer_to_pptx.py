"""Create an editable Suyana PowerPoint deck based on the Beamer output.

This module intentionally does not rasterize Beamer pages. The Suyana Beamer
deck contains a stable, generated structure, so the PPTX is rebuilt as native
PowerPoint content: editable text boxes, tables, shapes, connectors, and
shape-based charts populated from the pricing pipeline outputs. The chart
visuals are editable vector objects rather than embedded chart workbooks because
that produces the most reliable lightweight previews across PowerPoint,
Keynote, Quick Look, Google import, and browser previewers.

Usage
-----
    python beamer_to_pptx.py outputs/report_v7_20260429_1439
    python beamer_to_pptx.py outputs/report_v7_20260429_1439 --out deck.pptx
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# Suyana palette, mirrored from suyana_style.py / Beamer preamble.
SGREEN = "43A047"
SDARKGREEN = "2E7D32"
SBLACK = "141414"
SDARK = "111111"
SGRAY = "555555"
SMIDGRAY = "9E9E9E"
SLIGHTGRAY = "E0E0E0"
SRED = "D62728"

FONT_SCALE = 1.72
WIDE_W = Inches(13.333333)
WIDE_H = Inches(7.5)
CONTENT_L = Inches(0.92)
CONTENT_R = Inches(12.42)
TITLE_Y = Inches(0.36)
BODY_TOP = Inches(1.55)
FOOTER_Y = Inches(7.05)


SAFE_TEXT_REPLACEMENTS = {
    "–": "-",
    "—": "-",
    "·": " - ",
    "•": "-",
    "→": "->",
    "⇒": "=>",
    "≈": "~",
    "≤": "<=",
    "≥": ">=",
    "×": "x",
    "Σ": "sum",
    "λ": "lambda",
    "³": "3",
    "\u00a0": " ",
}

SLIDE_SUBTITLES = {
    "Agenda": "La historia avanza desde la cobertura y los datos hasta el diseno del indice, el contrato, la cotizacion y la ejecucion.",
    "Qué cubre el producto": "El producto define donde se cubre la sequia y como cada gatillo convierte estres hidrico en pago.",
    "Ventana de cobertura y propuesta de valor": "La ventana apunta al periodo critico del cultivo y reemplaza el ajuste de siniestro por liquidacion objetiva.",
    "Fuente de datos y cobertura temporal": "Treinta campanas de humedad de suelo ERA5 sostienen umbrales empiricos transparentes.",
    "Validación cuantitativa de la climatología": "Los controles automaticos verifican que la climatologia sea estable y auditable.",
    "Pipeline de construcción del índice": "Las observaciones diarias se transforman en anomalias, deficit estacional, umbrales y capas de pago.",
    "Cómo se calcula el índice de sequía acumulada": "El indice resume el deficit de humedad frente a lo normal durante la ventana asegurada.",
    "Ejemplo numérico paso a paso - campaña 2009, px_1 (Sur)": "El deficit severo de 2009 cruza P10, P5 y P1, mostrando como el indice se convierte en pago.",
    "Serie histórica de pérdidas anuales - 1996-2025": "Los pagos son poco frecuentes pero materiales, con el mayor evento por sequia severa simultanea.",
    "Curva de probabilidad anual de excedencia - portafolio": "La curva AEP traduce las perdidas a lenguaje de periodo de retorno para transferencia de riesgo.",
    "Estructura de capas del contrato": "Las capas pagan parcialmente eventos moderados y agotan limite en sequias raras y severas.",
    "El parámetro disparador K: cuatro lentes de lectura": "El mismo umbral se entiende como percentil, dato observado, frecuencia y regla economica.",
    "Cotización del portafolio - Trigo Argentina 1996-2025": "La cotizacion convierte perdida esperada y volatilidad en prima tecnica y luego comercial.",
    "Curva de probabilidad anual de excedencia por cultivo": "La curva muestra que las perdidas positivas son infrecuentes, pero la cola sigue definiendo precio.",
    "Próximos pasos": "El lanzamiento depende de validar exposicion, ampliar cultivos y cerrar poliza con reaseguro.",
}

SECTION_SUBTITLES = {
    "Producto y cobertura": "Primero se fija el cultivo, la geografia, la temporada y la promesa de pago.",
    "Datos y credibilidad": "Luego se demuestra que los datos y controles sostienen un indice auditable.",
    "Diseño del índice": "Despues se muestra como la humedad de suelo se convierte en gatillo.",
    "Diseño del contrato": "Luego el gatillo se traduce a capas, perdidas historicas y riesgo de portafolio.",
    "Resultados y cotización": "Finalmente el riesgo modelado se conecta con la prima cotizada.",
    "Próximos pasos": "El cierre ordena las decisiones operativas previas a la colocacion.",
}


def rgb(hex_value: str) -> RGBColor:
    hex_value = hex_value.strip("#")
    return RGBColor(
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )


def safe_text(value: object) -> str:
    """Normalize text to preview-safe XML/font content for generated PPTX."""
    text = str(value)
    for src, dst in SAFE_TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split()) if "\n" not in text else "\n".join(" ".join(line.split()) for line in text.splitlines())


def scaled_pt(size: int | float) -> Pt:
    return Pt(round(size * FONT_SCALE, 1))


def emu(value) -> int:
    return int(round(value))


def money(value: float, decimals: int = 0) -> str:
    return f"${value:,.{decimals}f}"


def clear_extended_attrs(path: Path) -> None:
    """Remove macOS metadata that can make generated artifacts preview poorly."""
    if shutil.which("xattr"):
        subprocess.run(["xattr", "-c", str(path)], check=False, capture_output=True)
        return
    if not hasattr(os, "listxattr") or not hasattr(os, "removexattr"):
        return
    try:
        attrs = os.listxattr(path)
    except OSError:
        return
    for attr in attrs:
        if attr.startswith("com.apple."):
            try:
                os.removexattr(path, attr)
            except OSError:
                pass


def add_text(
    slide,
    text: str,
    x,
    y,
    w,
    h,
    *,
    size: int = 18,
    bold: bool = False,
    color: str = SBLACK,
    align: PP_ALIGN | None = None,
    font: str = "Arial",
    margin: float = 0.02,
):
    text = safe_text(text)
    line_count = max(1, text.count("\n") + 1)
    min_h = Inches((size * FONT_SCALE * 1.25 * line_count) / 72 + margin * 2)
    if h < min_h:
        h = min_h
    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = scaled_pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_rich_line(slide, runs: list[tuple[str, str, bool]], x, y, w, h, size: int = 18):
    min_h = Inches((size * FONT_SCALE * 1.25) / 72 + 0.04)
    if h < min_h:
        h = min_h
    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    for value, color, bold in runs:
        run = p.add_run()
        run.text = safe_text(value)
        run.font.name = "Arial"
        run.font.size = scaled_pt(size)
        run.font.bold = bold
        run.font.color.rgb = rgb(color)
    return box


def add_line(slide, x1, y1, x2, y2, color: str = SGREEN, width: float = 1.2):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, emu(x1), emu(y1), emu(x2), emu(y2))
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    return line


def add_rect(
    slide,
    x,
    y,
    w,
    h,
    *,
    fill: str = "FFFFFF",
    line: str | None = None,
    radius_shape=MSO_SHAPE.RECTANGLE,
):
    shape = slide.shapes.add_shape(radius_shape, emu(x), emu(y), emu(w), emu(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(0.8)
    else:
        shape.line.fill.background()
    return shape


def add_bullets(slide, items: Iterable[str], x, y, w, h, *, size: int = 14, gap: float = 0.29):
    line_h = Inches((size * FONT_SCALE * 1.25) / 72 + 0.08)
    step = max(Inches(gap), line_h + Inches(0.12))
    yy = y
    for item in items:
        add_rich_line(
            slide,
            [("• ", SGREEN, True), (item, SBLACK, False)],
            x,
            yy,
            w,
            line_h,
            size=size,
        )
        yy += step


def add_numbered_items(slide, items: Iterable[tuple[str, str]], x, y, w, *, size: int = 14):
    yy = y
    line_h = Inches((size * FONT_SCALE * 1.25) / 72 + 0.22)
    for n, item in items:
        add_rich_line(
            slide,
            [(f"{n}. ", SGREEN, True), (item, SBLACK, False)],
            x,
            yy,
            w,
            line_h,
            size=size,
        )
        yy += line_h + Inches(0.3)


def add_table(slide, data: list[list[str]], x, y, w, h, *, font_size: int = 11):
    rows = len(data)
    cols = len(data[0])
    row_h = h / rows
    col_w = w / cols
    add_line(slide, x, y + row_h, x + w, y + row_h, SGREEN, 1.1)
    add_line(slide, x, y + h, x + w, y + h, SLIGHTGRAY, 0.7)
    for r, row in enumerate(data):
        yy = y + row_h * r + Inches(0.03)
        if r > 1:
            add_line(slide, x, y + row_h * r, x + w, y + row_h * r, "F0F0F0", 0.45)
        for c, value in enumerate(row):
            xx = x + col_w * c + Inches(0.04)
            align = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            color = SGREEN if r == 0 else SBLACK
            add_text(
                slide,
                value,
                xx,
                yy,
                col_w - Inches(0.08),
                row_h - Inches(0.04),
                size=font_size,
                bold=r == 0,
                color=color,
                align=align,
                margin=0.01,
            )


def add_header(slide, title: str, subtitle: str | None = None):
    add_line(slide, 0, Inches(0.08), WIDE_W, Inches(0.08), SGREEN, 1.6)
    add_text(slide, title, CONTENT_L, TITLE_Y, Inches(10.9), Inches(0.55), size=18, bold=True)
    if subtitle:
        add_text(slide, subtitle, CONTENT_L, Inches(0.96), Inches(11.15), Inches(0.54), size=11, color=SGRAY)
    add_line(slide, CONTENT_L, Inches(1.62), CONTENT_R, Inches(1.62), SMIDGRAY, 0.5)


def add_footer(slide, idx: int, total: int, product_name: str, month_year: str, confidential: str = "Confidencial"):
    add_line(slide, 0, FOOTER_Y, WIDE_W, FOOTER_Y, SMIDGRAY, 0.4)
    add_text(
        slide,
        f"{product_name} | {confidential} | {month_year}",
        Inches(0.08),
        Inches(7.09),
        Inches(8.0),
        Inches(0.34),
        size=7,
        color=SMIDGRAY,
    )
    add_text(
        slide,
        f"{idx}/{total}",
        Inches(12.25),
        Inches(7.09),
        Inches(0.8),
        Inches(0.34),
        size=7,
        color=SMIDGRAY,
        align=PP_ALIGN.RIGHT,
    )


def content_slide(prs, title: str, idx: int, total: int, meta: "DeckMeta", subtitle: str | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, title, subtitle or SLIDE_SUBTITLES.get(title, ""))
    add_footer(slide, idx, total, meta.product_name, meta.month_year)
    return slide


def dark_slide(prs, idx: int | None = None, total: int | None = None, meta: "DeckMeta" | None = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, WIDE_W, WIDE_H, fill=SDARK)
    if idx and total and meta:
        add_footer(slide, idx, total, meta.product_name, meta.month_year)
    return slide


@dataclass
class DeckMeta:
    product_name: str = "Producto Paramétrico de Sequía"
    month_year: str = "Abril 2026"
    region_label: str = "Argentina"


@dataclass
class PipelineData:
    pricing_quote: pd.DataFrame
    annual_losses: pd.DataFrame
    historical_losses: pd.DataFrame
    aep_curves: pd.DataFrame
    percentiles: pd.DataFrame
    config: dict


def load_pipeline_data(report_dir: Path) -> PipelineData:
    pricing_dir = report_dir.parents[1] if report_dir.name.startswith("report_") else report_dir
    if pricing_dir.name != "pricing":
        pricing_dir = Path(__file__).parent
    outputs = pricing_dir / "outputs"
    with (pricing_dir / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    return PipelineData(
        pricing_quote=pd.read_csv(outputs / "pricing_quote.csv"),
        annual_losses=pd.read_parquet(outputs / "aggregate_annual_losses.parquet"),
        historical_losses=pd.read_parquet(outputs / "historical_losses.parquet"),
        aep_curves=pd.read_parquet(outputs / "aep_curves.parquet"),
        percentiles=pd.read_parquet(outputs / "historical_percentiles.parquet"),
        config=config,
    )


def _plot_frame(slide, x, y, w, h, *, y_max: float, x_label: str, y_label: str):
    left = x + Inches(0.55)
    top = y + Inches(0.12)
    plot_w = w - Inches(0.78)
    plot_h = h - Inches(0.70)
    bottom = top + plot_h

    for tick in [0, y_max / 3, 2 * y_max / 3, y_max]:
        yy = bottom - int(plot_h * (tick / y_max))
        add_line(slide, left, yy, left + plot_w, yy, SLIGHTGRAY, 0.45)
        add_text(slide, f"{tick:.1f}", x, yy - Inches(0.11), Inches(0.45), Inches(0.22), size=5.5, color=SGRAY, align=PP_ALIGN.RIGHT)

    add_line(slide, left, top, left, bottom, SMIDGRAY, 0.7)
    add_line(slide, left, bottom, left + plot_w, bottom, SMIDGRAY, 0.7)
    return left, top, plot_w, plot_h, bottom


def add_annual_loss_chart(slide, data: PipelineData, x, y, w, h):
    df = data.annual_losses.sort_values("year")
    values = [v / 1_000_000 for v in df["loss_usd"]]
    years = [int(v) for v in df["year"]]
    y_max = 1.2
    left, top, plot_w, plot_h, bottom = _plot_frame(
        slide, x, y, w, h, y_max=y_max, x_label="Campaña", y_label="USD M"
    )
    n = len(values)
    slot = plot_w / n
    bar_w = slot * 0.58
    for i, value in enumerate(values):
        if value <= 0:
            continue
        bh = int(plot_h * min(value, y_max) / y_max)
        bx = left + int(slot * i + (slot - bar_w) / 2)
        by = bottom - bh
        fill = SGREEN if value < 1.0 else SDARKGREEN
        add_rect(slide, bx, by, int(bar_w), bh, fill=fill)
    for i, year in enumerate(years):
        if (i % 5 == 4) or i == n - 1:
            xx = left + int(slot * i)
            add_text(slide, str(year), xx - Inches(0.15), bottom + Inches(0.07), Inches(0.5), Inches(0.18), size=5.5, color=SGRAY, align=PP_ALIGN.CENTER)
    add_text(slide, "Pérdida anual", left + plot_w - Inches(1.25), top + Inches(0.05), Inches(1.2), Inches(0.22), size=8, color=SGREEN, align=PP_ALIGN.RIGHT)


def add_aep_chart(slide, data: PipelineData, x, y, w, h, *, crop: str = "Trigo"):
    df = data.aep_curves[data.aep_curves["crop"].eq(crop)].copy()
    df = df[df["exceedance_probability"].between(0.001, 0.30)]
    df = df.iloc[:: max(1, len(df) // 80)].sort_values("exceedance_probability")
    y_max = 1.2
    x_max = 30.0
    left, top, plot_w, plot_h, bottom = _plot_frame(
        slide, x, y, w, h, y_max=y_max, x_label="Probabilidad anual de excedencia (%)", y_label="USD M"
    )
    for tick in [10, 20, 30]:
        xx = left + int(plot_w * tick / x_max)
        add_text(slide, str(tick), xx - Inches(0.18), bottom + Inches(0.07), Inches(0.36), Inches(0.18), size=5.5, color=SGRAY, align=PP_ALIGN.CENTER)

    points: list[tuple[int, int]] = []
    for _, row in df.iterrows():
        px = left + int(plot_w * min(row["exceedance_probability"] * 100, x_max) / x_max)
        py = bottom - int(plot_h * min(row["loss_usd"] / 1_000_000, y_max) / y_max)
        points.append((px, py))

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        add_line(slide, x1, y1, x2, y2, SGREEN, 1.8)
    for px, py in points[:: max(1, len(points) // 12)]:
        marker = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            emu(px - Inches(0.035)),
            emu(py - Inches(0.035)),
            emu(Inches(0.07)),
            emu(Inches(0.07)),
        )
        marker.fill.solid()
        marker.fill.fore_color.rgb = rgb(SGREEN)
        marker.line.fill.background()
    add_text(slide, "AEP", left + plot_w - Inches(0.65), top + Inches(0.05), Inches(0.6), Inches(0.22), size=8, color=SGREEN, align=PP_ALIGN.RIGHT)


def add_pipeline_diagram(slide):
    labels = [
        ("Climatología\ndiaria ERA5\n1996-2025", "swvl1"),
        ("Anomalía\ndiaria\nDelta S_t", "anom_swvl1"),
        ("Anomalía\nacumulada\nventana", "15-sep -> 14-nov"),
        ("Percentiles\nhistóricos\nP1 / P5 / P10", "empírico"),
        ("Capa de\npago\n% límite", "USD/ha"),
    ]
    x0, y, bw, bh, gap = Inches(0.86), Inches(2.45), Inches(2.0), Inches(0.98), Inches(0.34)
    for i in range(len(labels) - 1):
        x1 = x0 + (bw + gap) * i + bw
        x2 = x0 + (bw + gap) * (i + 1)
        add_line(slide, x1, y + bh / 2, x2, y + bh / 2, SGREEN, 1.6)
    for i, (label, small) in enumerate(labels):
        x = x0 + (bw + gap) * i
        shape = add_rect(slide, x, y, bw, bh, fill="EEF7EE", line=SGREEN, radius_shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = shape.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = safe_text(label)
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = scaled_pt(11)
            run.font.color.rgb = rgb(SBLACK)
        add_text(slide, small, x, y + Inches(1.03), bw, Inches(0.24), size=9, color=SGRAY, align=PP_ALIGN.CENTER)


def add_layer_stack(slide):
    x, y, w = Inches(1.0), Inches(2.35), Inches(3.65)
    layers = [
        (y, Inches(1.35), "Capa atacante · P10", "20% límite/ha", "DDF0DD", SBLACK),
        (y + Inches(1.35), Inches(1.20), "Capa media · P5", "40% límite/ha", "8FCC92", SBLACK),
        (y + Inches(2.55), Inches(1.25), "Capa cola · P1", "100% límite/ha", SGREEN, "FFFFFF"),
    ]
    for yy, hh, label, sub, fill, text_color in layers:
        add_rect(slide, x, yy, w, hh, fill=fill)
        add_text(slide, label, x, yy + Inches(0.18), w, Inches(0.34), size=10.5, bold=True, color=text_color, align=PP_ALIGN.CENTER)
        add_text(slide, sub, x, yy + Inches(0.76), w, Inches(0.24), size=7.5, color=text_color, align=PP_ALIGN.CENTER)
    add_line(slide, x - Inches(0.18), y, x - Inches(0.18), y + Inches(3.95), SMIDGRAY, 1)
    add_text(slide, "Intensidad de sequía", x - Inches(0.58), y + Inches(1.4), Inches(0.38), Inches(1.1), size=8, color=SGRAY)


def create_editable_deck(report_dir: Path, out_path: Path | None = None) -> Path:
    report_dir = report_dir.resolve()
    out_path = (out_path or report_dir / "presentation.pptx").resolve()
    data = load_pipeline_data(report_dir)
    meta = DeckMeta()
    total = 23

    prs = Presentation()
    prs.slide_width = WIDE_W
    prs.slide_height = WIDE_H
    blank = prs.slide_layouts[6]

    # 1. Cover
    slide = dark_slide(prs)
    add_line(slide, Inches(1.1), Inches(3.2), Inches(12.2), Inches(3.2), SGREEN, 1.6)
    add_text(slide, "Seguro Paramétrico de Sequía", Inches(1.1), Inches(1.22), Inches(10.8), Inches(0.64), size=22, bold=True, color="FFFFFF")
    add_text(slide, "Trigo - Argentina", Inches(1.1), Inches(2.02), Inches(10.8), Inches(0.64), size=22, bold=True, color=SGREEN)
    add_text(slide, "Un producto transparente que conecta deficit de humedad de suelo con pagos auditables y una cotizacion comercial.\nRegiones Sur y Norte - Campana 2025", Inches(1.1), Inches(3.7), Inches(10.8), Inches(0.72), size=11, color=SMIDGRAY)
    add_text(slide, "Suyana - Producto Parametrico de Sequia v1.0.0 - Climatologia 1996-2025 - ERA5 swvl1", Inches(1.1), Inches(6.45), Inches(10.8), Inches(0.25), size=8, color=SGRAY)

    # 2. Agenda
    slide = content_slide(prs, "Agenda", 2, total, meta)
    agenda = [
        ("01", "Producto y cobertura"),
        ("02", "Datos y credibilidad"),
        ("03", "Diseño del índice"),
        ("04", "Diseño del contrato"),
        ("05", "Resultados y cotización"),
        ("06", "Próximos pasos"),
    ]
    for i, (num, label) in enumerate(agenda):
        y = Inches(2.05 + i * 0.68)
        add_text(slide, num, Inches(1.35), y, Inches(0.65), Inches(0.3), size=18, bold=True, color=SGREEN, align=PP_ALIGN.RIGHT)
        add_text(slide, label, Inches(2.25), y, Inches(6.5), Inches(0.3), size=18, color=SBLACK)

    # 3. Section
    section_slide(prs, "01", "Producto y cobertura")

    # 4. Coverage
    slide = content_slide(prs, "Qué cubre el producto", 4, total, meta)
    add_line(slide, Inches(0.98), Inches(2.05), Inches(6.0), Inches(2.05), SGREEN, 0.8)
    add_text(slide, "Cultivo y regiones", Inches(0.98), Inches(2.24), Inches(5.3), Inches(0.3), size=13, bold=True)
    add_bullets(slide, ["Cultivo: Trigo", "Region Sur: limite de perdida $600-$900 USD/ha", "Region Norte: limite de perdida $900-$1,200 USD/ha"], Inches(1.04), Inches(2.78), Inches(5.3), Inches(1.4), size=12, gap=0.43)
    add_line(slide, Inches(6.8), Inches(2.05), Inches(12.15), Inches(2.05), SGREEN, 0.8)
    add_text(slide, "Mecanismo de pago", Inches(6.8), Inches(2.24), Inches(5.3), Inches(0.3), size=13, bold=True)
    add_bullets(slide, ["Gatillo P10: 20% del limite regional por hectarea", "Gatillo P5: 40% del limite regional por hectarea", "Gatillo P1: 100% del limite regional por hectarea"], Inches(6.86), Inches(2.78), Inches(5.2), Inches(1.4), size=12, gap=0.43)
    add_text(slide, "Los porcentajes de pago se aplican sobre el limite midpoint de cada region (Sur $750 USD/ha - Norte $1,050 USD/ha) escalonado por capa. El area asegurada de referencia es 1,000 ha por pixel.", Inches(1.0), Inches(5.72), Inches(11.2), Inches(0.55), size=10, color=SMIDGRAY)

    # 5. Window and value prop
    slide = content_slide(prs, "Ventana de cobertura y propuesta de valor", 5, total, meta)
    add_line(slide, Inches(0.98), Inches(2.05), Inches(6.0), Inches(2.05), SGREEN, 0.8)
    add_text(slide, "Ventana de monitoreo", Inches(0.98), Inches(2.24), Inches(5.3), Inches(0.3), size=13, bold=True)
    add_bullets(slide, ["15 de septiembre al 14 de noviembre (61 dias)", "Cubre el periodo critico de macollaje y encanado del trigo", "Sin cruce de ano calendario"], Inches(1.04), Inches(2.78), Inches(5.3), Inches(1.4), size=12, gap=0.43)
    add_line(slide, Inches(6.8), Inches(2.05), Inches(12.15), Inches(2.05), SGREEN, 0.8)
    add_text(slide, "Propuesta de valor", Inches(6.8), Inches(2.24), Inches(5.3), Inches(0.3), size=13, bold=True)
    add_bullets(slide, ["Pago automatico y objetivo: humedad de suelo observada, sin ajuste de siniestro", "Escalonamiento transparente: tres capas con umbrales percentilicos historicos verificables"], Inches(6.86), Inches(2.78), Inches(5.2), Inches(1.5), size=12, gap=0.62)

    section_slide(prs, "02", "Datos y credibilidad")

    # 7. Data source
    slide = content_slide(prs, "Fuente de datos y cobertura temporal", 7, total, meta)
    add_table(slide, [
        ["Variable", "Fuente", "Período", "Resolución"],
        ["Humedad de suelo (capa 1)", "ERA5 swvl1", "1996-2025", "~25 km"],
        ["Anomalía acumulada", "Climatología interna", "1996-2025", "píxel"],
        ["Superficie asegurada", "Contrato (input)", "-", "1,000 ha/píxel"],
    ], Inches(1.05), Inches(2.12), Inches(11.05), Inches(1.75), font_size=12)
    add_text(slide, "El dataset cubre 30 campanas de trigo (1996-2025), con dos pixeles georreferenciados: px_1 = Zona A / Sur y px_2 = Zona B / Norte.", Inches(1.05), Inches(4.2), Inches(11.0), Inches(0.45), size=11, color=SMIDGRAY)
    add_text(slide, "La variable swvl1 es la humedad volumetrica de la primera capa de suelo (0-7 cm) del reanalisis ERA5 de ECMWF, procesada a resolucion diaria en modo pixel-level sin interpolacion espacial adicional.", Inches(1.05), Inches(5.02), Inches(11.0), Inches(0.65), size=11, color=SMIDGRAY)

    # 8. Climatology QA
    slide = content_slide(prs, "Validación cuantitativa de la climatología", 8, total, meta)
    add_text(slide, "Estadistica de la climatologia de referencia", Inches(0.98), Inches(1.95), Inches(5.5), Inches(0.3), size=13, bold=True)
    add_table(slide, [
        ["Métrica", "Valor"],
        ["Registros (días x píxeles)", "732"],
        ["Media humedad", "0.3005 m3/m3"],
        ["Desvío estándar", "0.0092 m3/m3"],
        ["Percentil 5%", "0.2857 m3/m3"],
        ["Percentil 95%", "0.3153 m3/m3"],
    ], Inches(0.98), Inches(2.45), Inches(5.5), Inches(2.65), font_size=10.5)
    add_text(slide, "Contratos de datos: todos aprobados", Inches(7.05), Inches(1.95), Inches(5.0), Inches(0.3), size=13, bold=True)
    add_bullets(slide, ["baseline_climatology: columnas completas", "cumulative_anomalies: sin nulos (60 registros)", "historical_percentiles: orden P1 < P5 < P10 verificado", "historical_losses: perdidas >= 0 en todos los anos", "pricing_quote: consistencia prima tecnica/comercial"], Inches(7.05), Inches(2.45), Inches(5.2), Inches(2.6), size=10.8, gap=0.43)

    section_slide(prs, "03", "Diseño del índice")

    # 10. Pipeline diagram
    slide = content_slide(prs, "Pipeline de construcción del índice", 10, total, meta)
    add_pipeline_diagram(slide)
    add_text(slide, "La climatologia de referencia se calcula como la media historica de cada dia del ano sobre 1996-2025. La anomalia diaria es la diferencia entre el valor observado y esa media; su suma dentro de la ventana de cultivo constituye el indice de sequia acumulada.", Inches(1.0), Inches(5.0), Inches(11.3), Inches(0.85), size=11, color=SMIDGRAY)

    # 11. Index interpretation
    slide = content_slide(prs, "Cómo se calcula el índice de sequía acumulada", 11, total, meta)
    add_text(slide, "Idea central", Inches(1.0), Inches(2.05), Inches(10.5), Inches(0.35), size=13, bold=True, color=SGREEN)
    add_text(slide, "Cada dia se compara la humedad observada contra lo normal para ese pixel y esa fecha. Luego se acumula el deficit durante la ventana asegurada.", Inches(1.0), Inches(2.58), Inches(11.2), Inches(1.05), size=18, bold=True, color=SBLACK)
    add_bullets(slide, ["Valor mas negativo: sequia mas intensa durante la ventana", "Umbrales P10, P5 y P1: niveles de rareza historica calculados por pixel", "Pago: se activa cuando el deficit acumulado cruza el umbral de la capa correspondiente"], Inches(1.05), Inches(4.35), Inches(11.0), Inches(1.5), size=12, gap=0.48)

    # 12. Numerical example
    slide = content_slide(prs, "Ejemplo numérico paso a paso - campaña 2009, px_1 (Sur)", 12, total, meta)
    add_line(slide, Inches(1.0), Inches(1.95), Inches(12.15), Inches(1.95), SGREEN, 0.8)
    add_text(slide, "Paso 1 - Anomalia acumulada observada", Inches(1.0), Inches(2.18), Inches(11), Inches(0.3), size=13, bold=True)
    add_text(slide, "El pipeline registra para px_1 en 2009 un deficit de -0.8132 m3/m3-dias.", Inches(1.0), Inches(2.65), Inches(11), Inches(0.3), size=11)
    add_text(slide, "Paso 2 - Comparacion contra umbrales empiricos", Inches(1.0), Inches(3.18), Inches(11), Inches(0.3), size=13, bold=True)
    add_table(slide, [["Umbral", "Valor empirico", "Deficit 2009", "Activado"], ["P10", "-0.5780", "-0.8132", "Si"], ["P5", "-0.6612", "-0.8132", "Si"], ["P1", "-0.7794", "-0.8132", "Si"]], Inches(1.1), Inches(3.7), Inches(8.5), Inches(1.55), font_size=10.5)
    add_text(slide, "Paso 3 - Capa activada: P1 => pago = 100% del limite", Inches(1.0), Inches(5.55), Inches(11), Inches(0.3), size=13, bold=True)
    add_text(slide, "Limite midpoint Sur: $750 USD/ha - Area del pixel: 1,000 ha - Pago total px_1, campana 2009: $750,000 USD.", Inches(1.0), Inches(6.0), Inches(11.2), Inches(0.45), size=10.5)

    section_slide(prs, "04", "Diseño del contrato")

    # 14. Annual loss chart
    slide = content_slide(prs, "Serie histórica de pérdidas anuales - 1996-2025", 14, total, meta)
    add_annual_loss_chart(slide, data, Inches(1.0), Inches(1.95), Inches(11.35), Inches(4.2))
    add_text(slide, "Lectura: seis de 30 campanas generaron pago. El pico de $1.20 M ocurrio en 2009 por activacion P1 simultanea en Sur y Norte; el riesgo se concentra entre 2004 y 2017.", Inches(1.0), Inches(6.3), Inches(11.2), Inches(0.42), size=10, color=SGRAY)

    # 15. AEP portfolio
    slide = content_slide(prs, "Curva de probabilidad anual de excedencia - portafolio", 15, total, meta)
    add_aep_chart(slide, data, Inches(1.0), Inches(1.95), Inches(11.35), Inches(4.2))
    add_text(slide, "Eje X: probabilidad anual de excedencia (%). Eje Y: perdida (USD M). La perdida media anual converge en $0.10 M; exceder $0.40 M ocurre cerca de 10% anual.", Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.42), size=10, color=SGRAY)

    # 16. Contract layers
    slide = content_slide(prs, "Estructura de capas del contrato", 16, total, meta)
    add_text(slide, "Esquema de capas", Inches(1.0), Inches(1.95), Inches(4.7), Inches(0.3), size=13, bold=True)
    add_layer_stack(slide)
    add_text(slide, "Parametros por region", Inches(6.6), Inches(1.95), Inches(5.5), Inches(0.3), size=13, bold=True)
    add_table(slide, [["", "Sur", "Norte"], ["Limite min (USD/ha)", "600", "900"], ["Limite mid (USD/ha)", "750", "1,050"], ["Limite max (USD/ha)", "900", "1,200"], ["P10 pago", "20%", "20%"], ["P5 pago", "40%", "40%"], ["P1 pago", "100%", "100%"]], Inches(6.6), Inches(2.45), Inches(5.35), Inches(3.25), font_size=10)

    # 17. Four lenses
    slide = content_slide(prs, "El parámetro disparador K: cuatro lentes de lectura", 17, total, meta)
    add_table(slide, [
        ["Lente", "Interpretacion del umbral K (ej. P10)"],
        ["Mecanica", "Percentil 10 de la distribucion empirica de anomalia acumulada; se activa cuando la sequia supera ese nivel de rareza historica."],
        ["En el dato", "K_P10 = -0.578 m3/m3·d (Sur) y -0.432 m3/m3·d (Norte), calibrado sobre 30 temporadas de ERA5."],
        ["En el tiempo", "En 30 anos, el umbral P10 se activo 6 veces; frecuencia observada = 10%, consistente con el diseno."],
        ["En la economia", "Cada activacion P10 genera $150 USD/ha (Sur) o $210 USD/ha (Norte); P1 genera $750 o $1,050 USD/ha."],
    ], Inches(1.0), Inches(1.95), Inches(11.35), Inches(4.15), font_size=8.5)
    add_rich_line(slide, [("Reasegurador: periodo de retorno. ", SGREEN, True), ("Cedente: extension de cartera.", SGRAY, False)], Inches(1.0), Inches(6.2), Inches(10.8), Inches(0.3), size=11)

    section_slide(prs, "05", "Resultados y cotización")

    # 19. Pricing
    slide = content_slide(prs, "Cotización del portafolio - Trigo Argentina 1996-2025", 19, total, meta)
    combined = data.pricing_quote[data.pricing_quote["crop"].eq("Combined")].iloc[0]
    add_table(slide, [["Cultivo", "Perdida media anual", "Desvio estandar", "Prima tecnica", "Prima comercial"], ["Trigo (combinado)", money(combined.AAL_usd), money(combined.StdDev_usd), money(combined.technical_premium_usd), money(combined.commercial_premium_usd)]], Inches(0.98), Inches(2.05), Inches(11.45), Inches(1.1), font_size=9.5)
    add_text(slide, "Todas las cifras estan en USD. La prima tecnica carga perdida esperada y volatilidad; la prima comercial agrega el margen objetivo.", Inches(1.0), Inches(3.48), Inches(11.4), Inches(0.55), size=11, color=SMIDGRAY)
    add_text(slide, "Lectura de precio", Inches(1.0), Inches(4.45), Inches(10.8), Inches(0.32), size=13, bold=True)
    add_text(slide, "La prima comercial de $650,957 equivale a 6.26x la perdida media anual. Ese multiple refleja que el riesgo es infrecuente, pero con cola material.", Inches(1.0), Inches(4.95), Inches(11.2), Inches(0.9), size=13, color=SBLACK)

    # 20. AEP per crop
    slide = content_slide(prs, "Curva de probabilidad anual de excedencia por cultivo", 20, total, meta)
    add_aep_chart(slide, data, Inches(1.0), Inches(1.95), Inches(11.35), Inches(4.2))
    add_text(slide, "Lectura: trigo excede perdida positiva cerca de 22% anual, consistente con seis activaciones en 30 temporadas. El maximo de cartera queda cerca de 3%.", Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.42), size=10, color=SGRAY)

    section_slide(prs, "06", "Próximos pasos")

    # 22. Next steps
    slide = content_slide(prs, "Próximos pasos", 22, total, meta)
    add_line(slide, Inches(1.0), Inches(1.95), Inches(12.1), Inches(1.95), SGREEN, 0.8)
    add_numbered_items(slide, [
        ("1", "Validacion de cartera: confirmar superficie asegurada por zona y limites definitivos para ajustar AAL y prima comercial a la exposicion real."),
        ("2", "Incorporacion de cultivos: extender a Soja Sur, Maiz Sur, Soja Norte y Maiz Norte con ventanas ya configuradas."),
        ("3", "Emision y reaseguro: cerrar contrato definitivo, retenciones y colocacion bajo la estructura de capas."),
    ], Inches(1.05), Inches(2.45), Inches(11.0), size=13)

    # 23. Closing
    slide = dark_slide(prs)
    add_line(slide, Inches(1.1), Inches(5.3), Inches(12.2), Inches(5.3), SGREEN, 1.6)
    add_text(slide, "Gracias", Inches(0.0), Inches(2.7), WIDE_W, Inches(0.6), size=28, bold=True, color="FFFFFF", align=PP_ALIGN.CENTER)
    add_text(slide, "El producto queda listo para validar exposicion, cerrar redaccion de poliza y estructurar reaseguro.", Inches(1.35), Inches(3.7), Inches(10.6), Inches(0.4), size=11, color=SMIDGRAY, align=PP_ALIGN.CENTER)
    add_text(slide, "Suyana · Seguro Paramétrico de Sequía · Trigo Argentina", Inches(0.0), Inches(4.22), WIDE_W, Inches(0.3), size=9, color=SMIDGRAY, align=PP_ALIGN.CENTER)
    add_text(slide, "Producto Parametrico de Sequia v1.0.0 - Climatologia ERA5 1996-2025 - 13 de marzo de 2026 - Confidencial", Inches(1.1), Inches(5.85), Inches(11.1), Inches(0.3), size=8, color=SGRAY, align=PP_ALIGN.CENTER)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    clear_extended_attrs(out_path)
    return out_path


def section_slide(prs, number: str, label: str):
    slide = dark_slide(prs)
    add_text(slide, number, Inches(1.05), Inches(2.45), Inches(2.1), Inches(1.1), size=48, bold=True, color=SGREEN)
    add_line(slide, Inches(1.08), Inches(3.88), Inches(12.05), Inches(3.88), SGREEN, 1.2)
    add_text(slide, label, Inches(1.08), Inches(4.25), Inches(9.8), Inches(0.72), size=20, bold=True, color="FFFFFF")
    add_text(slide, SECTION_SUBTITLES.get(label, ""), Inches(1.1), Inches(5.18), Inches(10.9), Inches(0.68), size=11, color=SMIDGRAY)
    return slide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path, help="Generated report folder containing presentation.tex/.pdf")
    parser.add_argument("--out", type=Path, help="Output .pptx path")
    args = parser.parse_args()
    pptx = create_editable_deck(args.report_dir, args.out)
    print(f"Wrote editable PPTX: {pptx}")


if __name__ == "__main__":
    main()
