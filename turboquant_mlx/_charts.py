from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Tuple

CHART_FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
GRID_COLOR = "#E2E8F0"
AXIS_COLOR = "#CBD5E1"
TITLE_COLOR = "#0F172A"
SUBTITLE_COLOR = "#475569"
TICK_COLOR = "#64748B"
LABEL_COLOR = "#334155"
SERIES_PALETTE = ["#2463EB", "#DC2626", "#059669", "#7C3AED", "#EA580C", "#0F766E"]


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def format_bytes(value: float) -> str:
    if value <= 0:
        return "0 B"
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{int(value)} B"


def _nice_step(span: float) -> float:
    if span <= 0:
        return 1.0
    exponent = 10 ** (len(str(int(span))) - 1)
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * exponent / 10
        if span / step <= 6:
            return step
    return span / 4


def _domain(values: List[float], *, zero_baseline: bool, pad_ratio: float = 0.08) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if zero_baseline:
        lo = min(lo, 0.0)
    if lo == hi:
        return (lo - 1.0, hi + 1.0) if lo != 0 else (0.0, 1.0)
    span = hi - lo
    pad = span * pad_ratio
    if zero_baseline and lo >= 0:
        return 0.0, hi + pad
    return lo - pad, hi + pad


def _yticks(lo: float, hi: float, count: int = 5) -> List[float]:
    if hi == lo:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count)]


def _legend(parts: List[str], items: List[Tuple[str, str]], *, x: float, y: float, max_width: float) -> None:
    cursor_x, cursor_y = x, y
    row_height = 22
    for color, name in items:
        approx_width = 28 + len(name) * 7
        if cursor_x + approx_width > x + max_width and cursor_x != x:
            cursor_x = x
            cursor_y += row_height
        parts.append(
            f'<rect x="{cursor_x}" y="{cursor_y - 11}" width="14" height="14" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cursor_x + 22}" y="{cursor_y}" font-size="13" font-family="{CHART_FONT}" fill="{LABEL_COLOR}">{name}</text>'
        )
        cursor_x += approx_width + 8


def _frame(
    title: str,
    subtitle: str,
    *,
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    x_label: str = "",
    y_label: str = "",
) -> List[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="{left}" y="42" font-size="22" font-weight="600" font-family="{CHART_FONT}" fill="{TITLE_COLOR}">{title}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{left}" y="66" font-size="13" font-family="{CHART_FONT}" fill="{SUBTITLE_COLOR}">{subtitle}</text>'
        )
    if x_label:
        parts.append(
            f'<text x="{left + (width - left - right) / 2}" y="{height - 12}" '
            f'text-anchor="middle" font-size="13" font-family="{CHART_FONT}" fill="{LABEL_COLOR}">{x_label}</text>'
        )
    if y_label:
        cy = top + (height - top - bottom) / 2
        parts.append(
            f'<text x="20" y="{cy}" text-anchor="middle" font-size="13" '
            f'font-family="{CHART_FONT}" fill="{LABEL_COLOR}" transform="rotate(-90 20 {cy})">{y_label}</text>'
        )
    return parts


def bar_chart_svg(
    title: str,
    labels: List[str],
    values: List[float],
    *,
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    width: int = 900,
    height: int = 540,
    color: str = "#2463EB",
    formatter: Callable[[float], str] = _format_float,
    reference_line: Optional[float] = None,
    reference_label: str = "",
) -> str:
    if not values:
        raise ValueError("bar chart requires at least one value")
    left, right, top, bottom = 90, 40, 90, 96
    chart_width = width - left - right
    chart_height = height - top - bottom
    origin_y = top + chart_height
    gap = 24
    bar_width = max(24, (chart_width - gap * (len(values) - 1)) / len(values))
    domain_values = list(values) + ([reference_line] if reference_line is not None else [])
    lo, hi = _domain(domain_values, zero_baseline=True)

    parts = _frame(title, subtitle, width=width, height=height, left=left, right=right, top=top, bottom=bottom, x_label=x_label, y_label=y_label)

    for tick in _yticks(lo, hi):
        y = origin_y - chart_height * (tick - lo) / (hi - lo)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="{CHART_FONT}" fill="{TICK_COLOR}">{formatter(tick)}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{origin_y}" x2="{width - right}" y2="{origin_y}" stroke="{AXIS_COLOR}" stroke-width="1.5"/>')

    if reference_line is not None:
        ref_y = origin_y - chart_height * (reference_line - lo) / (hi - lo)
        parts.append(
            f'<line x1="{left}" y1="{ref_y:.1f}" x2="{width - right}" y2="{ref_y:.1f}" '
            f'stroke="#94A3B8" stroke-width="1.2" stroke-dasharray="6,4"/>'
        )
        if reference_label:
            parts.append(
                f'<text x="{width - right - 6:.1f}" y="{ref_y - 6:.1f}" text-anchor="end" '
                f'font-size="11" font-family="{CHART_FONT}" fill="#475569">{reference_label}</text>'
            )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * (bar_width + gap)
        bar_h = chart_height * (value - lo) / (hi - lo)
        y = origin_y - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" rx="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-size="12" font-weight="600" font-family="{CHART_FONT}" fill="{TITLE_COLOR}">{formatter(value)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{origin_y + 22}" text-anchor="middle" '
            f'font-size="13" font-family="{CHART_FONT}" fill="{LABEL_COLOR}">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def line_chart_svg(
    title: str,
    series: Mapping[str, List[float]],
    x_labels: List[str],
    *,
    subtitle: str = "",
    x_label: str = "Context length (tokens)",
    y_label: str = "",
    width: int = 960,
    height: int = 560,
    formatter: Callable[[float], str] = _format_float,
    zero_baseline: bool = False,
    show_point_values: bool = True,
) -> str:
    if not series:
        raise ValueError("line chart requires series")
    left, right, top, bottom = 96, 40, 90, 110
    chart_width = width - left - right
    chart_height = height - top - bottom
    origin_y = top + chart_height

    flat = [v for vals in series.values() for v in vals]
    lo, hi = _domain(flat, zero_baseline=zero_baseline)
    span = hi - lo or 1.0

    parts = _frame(title, subtitle, width=width, height=height, left=left, right=right, top=top, bottom=bottom, x_label=x_label, y_label=y_label)

    for tick in _yticks(lo, hi):
        y = origin_y - chart_height * (tick - lo) / span
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="{CHART_FONT}" fill="{TICK_COLOR}">{formatter(tick)}</text>'
        )

    n = len(x_labels)
    xs = [left + chart_width / 2] if n == 1 else [left + i * (chart_width / (n - 1)) for i in range(n)]
    for x, label in zip(xs, x_labels):
        parts.append(
            f'<text x="{x:.1f}" y="{origin_y + 22}" text-anchor="middle" font-size="12" '
            f'font-family="{CHART_FONT}" fill="{LABEL_COLOR}">{label}</text>'
        )

    legend_items: List[Tuple[str, str]] = []
    palette = SERIES_PALETTE
    for color, (name, values) in zip(palette, series.items()):
        if not values:
            continue
        usable_xs = xs[: len(values)]
        pts = [(x, origin_y - chart_height * (v - lo) / span) for x, v in zip(usable_xs, values)]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
        for (x, y), v in zip(pts, values):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            if show_point_values:
                parts.append(
                    f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="11" '
                    f'font-family="{CHART_FONT}" fill="{color}">{formatter(v)}</text>'
                )
        legend_items.append((color, name))

    _legend(parts, legend_items, x=left, y=height - 36, max_width=width - left - right)
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_chart_svg(
    title: str,
    points: List[dict],
    *,
    subtitle: str = "",
    x_label: str = "Cache memory (bytes)",
    y_label: str = "Quality (mean F1)",
    width: int = 960,
    height: int = 580,
    color_map: Optional[Mapping[str, str]] = None,
) -> str:
    if not points:
        raise ValueError("scatter chart requires at least one point")
    left, right, top, bottom = 96, 40, 90, 120
    chart_width = width - left - right
    chart_height = height - top - bottom
    origin_y = top + chart_height
    color_map = color_map or {
        "standard": "#2463EB",
        "preset_3p5_qjl": "#059669",
        "preset_2p5_qjl": "#EA580C",
        "core_4bit": "#7C3AED",
    }

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x_lo, x_hi = _domain(xs, zero_baseline=True)
    y_lo, y_hi = _domain(ys, zero_baseline=False)
    x_span = x_hi - x_lo or 1.0
    y_span = y_hi - y_lo or 1.0

    parts = _frame(title, subtitle, width=width, height=height, left=left, right=right, top=top, bottom=bottom, x_label=x_label, y_label=y_label)

    for tick in _yticks(y_lo, y_hi):
        y = origin_y - chart_height * (tick - y_lo) / y_span
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="{CHART_FONT}" fill="{TICK_COLOR}">{tick:.3f}</text>'
        )
    for tick in _yticks(x_lo, x_hi):
        x = left + chart_width * (tick - x_lo) / x_span
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{origin_y}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{origin_y + 22}" text-anchor="middle" font-size="11" '
            f'font-family="{CHART_FONT}" fill="{TICK_COLOR}">{format_bytes(tick)}</text>'
        )

    used = []
    for idx, point in enumerate(points):
        x = left + chart_width * (point["x"] - x_lo) / x_span
        y = origin_y - chart_height * (point["y"] - y_lo) / y_span
        c = color_map.get(point["mode_slug"], "#334155")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{c}" opacity="0.9"/>')
        offset_y = -10 if idx % 2 == 0 else 18
        parts.append(
            f'<text x="{x + 10:.1f}" y="{y + offset_y:.1f}" font-size="11" '
            f'font-family="{CHART_FONT}" fill="{TITLE_COLOR}">{point["label"]}</text>'
        )
        used.append(point["mode_slug"])

    legend_items = [(color_map.get(slug, "#334155"), slug) for slug in dict.fromkeys(used)]
    _legend(parts, legend_items, x=left, y=height - 36, max_width=width - left - right)
    parts.append("</svg>")
    return "\n".join(parts)
