"""
lineage_visualizer.py
Generates an interactive SVG/HTML data lineage diagram showing:
  Input columns → Transformation logic → Output columns
No external graph library required — pure HTML + inline SVG.
"""

from __future__ import annotations
import html
import textwrap
from typing import Optional


def build_lineage_html(
    input_columns: list[str],
    target_columns: list[str],
    transformation_logic: dict[str, str],
    crosswalk_tables: Optional[list[dict]] = None,
) -> str:
    """
    Build a full HTML page containing an SVG lineage diagram.
    Suitable for `st.components.v1.html(build_lineage_html(...), height=...)`.
    """

    BOX_W = 180
    BOX_H = 36
    GAP_Y = 14
    COL_SPACING = 320
    PADDING_TOP = 60
    PADDING_LEFT = 40

    n_in = len(input_columns)
    n_out = len(target_columns)
    n_cw = len(crosswalk_tables) if crosswalk_tables else 0

    # Total height needed
    max_rows = max(n_in, n_out, n_cw, 1)
    total_h = PADDING_TOP + max_rows * (BOX_H + GAP_Y) + 80

    # Column x positions
    x_in = PADDING_LEFT
    x_mid = PADDING_LEFT + COL_SPACING           # transformation box (centre label, no boxes)
    x_cw = x_mid + 10                             # crosswalk tables (if any) go below mid
    x_out = PADDING_LEFT + COL_SPACING * 2

    total_w = x_out + BOX_W + PADDING_LEFT

    # ── Helpers ────────────────────────────────────────────────────────

    def box_y(index: int) -> int:
        return PADDING_TOP + index * (BOX_H + GAP_Y)

    def box_cy(index: int) -> int:
        return box_y(index) + BOX_H // 2

    def esc(s: str) -> str:
        return html.escape(str(s))

    def truncate(s: str, n: int = 22) -> str:
        return s[:n] + "…" if len(s) > n else s

    # ── Build SVG elements ─────────────────────────────────────────────

    rects_in: list[str] = []
    rects_out: list[str] = []
    rects_cw: list[str] = []
    arrows: list[str] = []
    labels: list[str] = []

    # Input column boxes
    for i, col in enumerate(input_columns):
        y = box_y(i)
        cy = box_cy(i)
        rects_in.append(
            f'<rect x="{x_in}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
            f'class="box-in" data-col="{esc(col)}" />'
        )
        rects_in.append(
            f'<text x="{x_in + BOX_W // 2}" y="{y + BOX_H // 2 + 5}" '
            f'class="box-label">{esc(truncate(col))}</text>'
        )

    # Target column boxes
    for i, col in enumerate(target_columns):
        y = box_y(i)
        cy = box_cy(i)
        rects_out.append(
            f'<rect x="{x_out}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
            f'class="box-out" data-col="{esc(col)}" />'
        )
        rects_out.append(
            f'<text x="{x_out + BOX_W // 2}" y="{y + BOX_H // 2 + 5}" '
            f'class="box-label">{esc(truncate(col))}</text>'
        )

    # Crosswalk boxes (right side of input, offset vertically)
    cw_start_y_offset = max(n_in, 1) * (BOX_H + GAP_Y) + PADDING_TOP + 20
    if crosswalk_tables:
        for i, cw in enumerate(crosswalk_tables):
            y = cw_start_y_offset + i * (BOX_H + GAP_Y)
            cy = y + BOX_H // 2
            rects_cw.append(
                f'<rect x="{x_in}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                f'class="box-cw" />'
            )
            rects_cw.append(
                f'<text x="{x_in + BOX_W // 2}" y="{y + BOX_H // 2 + 5}" '
                f'class="box-label cw-label">{esc(truncate(cw["name"]))}</text>'
            )
            # Arrow from crosswalk to output
            src_x = x_in + BOX_W
            dst_x = x_out
            mid_x = (src_x + dst_x) // 2
            arrows.append(
                f'<path d="M{src_x},{cy} C{mid_x},{cy} {mid_x},{PADDING_TOP + 0 * (BOX_H + GAP_Y) + BOX_H // 2} '
                f'{dst_x},{PADDING_TOP + 0 * (BOX_H + GAP_Y) + BOX_H // 2}" '
                f'class="arrow cw-arrow" />'
            )

    # Arrows: each input col → each output col
    # Smart: if 1-to-1 mapping (same count), draw paired lines; else fan from each input to each output
    arrow_opacity = "0.35" if n_in * n_out > 20 else "0.5"
    if n_in == n_out:
        for i in range(n_in):
            src_x = x_in + BOX_W
            src_y = box_cy(i)
            dst_x = x_out
            dst_y = box_cy(i)
            mid_x = (src_x + dst_x) // 2
            arrows.append(
                f'<path d="M{src_x},{src_y} C{mid_x},{src_y} {mid_x},{dst_y} {dst_x},{dst_y}" '
                f'class="arrow main-arrow" opacity="{arrow_opacity}" />'
            )
    else:
        # Fan: each input → each output
        for i in range(n_in):
            for j in range(n_out):
                src_x = x_in + BOX_W
                src_y = box_cy(i)
                dst_x = x_out
                dst_y = box_cy(j)
                mid_x = (src_x + dst_x) // 2
                arrows.append(
                    f'<path d="M{src_x},{src_y} C{mid_x},{src_y} {mid_x},{dst_y} {dst_x},{dst_y}" '
                    f'class="arrow main-arrow" opacity="0.2" />'
                )

    # Transformation logic labels (mid-point, one per output col)
    for j, col in enumerate(target_columns):
        logic = transformation_logic.get(col, "")
        short_logic = truncate(logic, 30) if logic else "pass-through"
        mid_y = box_cy(j)
        mid_x_centre = x_in + BOX_W + (x_out - x_in - BOX_W) // 2
        labels.append(
            f'<text x="{mid_x_centre}" y="{mid_y - 6}" class="logic-label">{esc(short_logic)}</text>'
        )
        labels.append(
            f'<line x1="{mid_x_centre - 40}" y1="{mid_y}" x2="{mid_x_centre + 40}" y2="{mid_y}" '
            f'class="logic-line" />'
        )

    # Column headers
    header_y = PADDING_TOP - 20
    headers = [
        f'<text x="{x_in + BOX_W // 2}" y="{header_y}" class="section-header">📥 Input</text>',
        f'<text x="{x_in + BOX_W + (x_out - x_in - BOX_W) // 2}" y="{header_y}" class="section-header">⚙️ Transform</text>',
        f'<text x="{x_out + BOX_W // 2}" y="{header_y}" class="section-header">📤 Output</text>',
    ]
    if crosswalk_tables:
        headers.append(
            f'<text x="{x_in + BOX_W // 2}" y="{cw_start_y_offset - 10}" class="section-header cw-section">🔗 Crosswalks</text>'
        )

    # Adjust total height for crosswalks
    if crosswalk_tables:
        total_h = max(total_h, cw_start_y_offset + n_cw * (BOX_H + GAP_Y) + 40)

    # ── Assemble SVG ───────────────────────────────────────────────────

    svg_body = "\n".join(
        arrows + rects_in + rects_out + rects_cw + labels + headers
    )

    css = """
    <style>
      body { margin: 0; background: #f8fafc; font-family: 'Inter', sans-serif; }
      svg { display: block; margin: 0 auto; }

      .box-in {
        fill: #dbeafe; stroke: #3b82f6; stroke-width: 1.5;
        filter: drop-shadow(0 2px 4px rgba(59,130,246,0.2));
        cursor: pointer; transition: fill 0.2s;
      }
      .box-in:hover { fill: #93c5fd; }

      .box-out {
        fill: #dcfce7; stroke: #22c55e; stroke-width: 1.5;
        filter: drop-shadow(0 2px 4px rgba(34,197,94,0.2));
        cursor: pointer; transition: fill 0.2s;
      }
      .box-out:hover { fill: #86efac; }

      .box-cw {
        fill: #fef9c3; stroke: #eab308; stroke-width: 1.5;
        filter: drop-shadow(0 2px 4px rgba(234,179,8,0.2));
      }

      .box-label {
        text-anchor: middle; font-size: 12px; font-weight: 600;
        fill: #1e293b; pointer-events: none;
      }
      .cw-label { fill: #78350f; }

      .main-arrow {
        fill: none; stroke: #6366f1; stroke-width: 1.5;
        marker-end: url(#arrowhead);
      }
      .cw-arrow {
        fill: none; stroke: #eab308; stroke-width: 1.5; stroke-dasharray: 6 3;
        marker-end: url(#arrowhead-cw);
      }

      .logic-label {
        text-anchor: middle; font-size: 10px; fill: #64748b; font-style: italic;
      }
      .logic-line {
        stroke: #cbd5e1; stroke-width: 1; stroke-dasharray: 4 2;
      }

      .section-header {
        text-anchor: middle; font-size: 13px; font-weight: 700;
        fill: #334155; letter-spacing: 0.02em;
      }
      .cw-section { fill: #92400e; }
    </style>
    """

    defs = f"""
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
      </marker>
      <marker id="arrowhead-cw" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#eab308" />
      </marker>
    </defs>
    """

    svg = (
        f'<svg width="{total_w}" height="{total_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#f8fafc; border-radius:12px;">'
        f"{defs}{svg_body}</svg>"
    )

    return f"<!DOCTYPE html><html><head>{css}</head><body>{svg}</body></html>"
