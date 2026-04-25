from __future__ import annotations

from typing import Callable, List, Mapping


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def bar_chart_svg(
    title: str,
    labels: List[str],
    values: List[float],
    *,
    subtitle: str = "",
    width: int = 900,
    height: int = 520,
    color: str = "#2463EB",
    formatter: Callable[[float], str] = _format_float,
) -> str:
    if not values:
        raise ValueError("bar chart requires at least one value")
    max_value = max(values) or 1.0
    left, right, top, bottom = 90, 40, 110, 110
    chart_width = width - left - right
    chart_height = height - top - bottom
    gap = 24
    bar_width = max(24, (chart_width - gap * (len(values) - 1)) / len(values))
    origin_y = top + chart_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        f'<text x="{left}" y="44" font-size="28" font-family="Menlo, Monaco, monospace" fill="#0F172A">{title}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{left}" y="72" font-size="14" font-family="Menlo, Monaco, monospace" fill="#475569">{subtitle}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{origin_y}" x2="{width-right}" y2="{origin_y}" stroke="#CBD5E1" stroke-width="2"/>')

    for i in range(5):
        tick_value = max_value * i / 4
        y = origin_y - chart_height * i / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-size="12" font-family="Menlo, Monaco, monospace" fill="#64748B">{formatter(tick_value)}</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * (bar_width + gap)
        bar_height = 0 if max_value == 0 else chart_height * (value / max_value)
        y = origin_y - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="6" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_width/2}" y="{y-10}" text-anchor="middle" font-size="12" font-family="Menlo, Monaco, monospace" fill="#0F172A">{formatter(value)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width/2}" y="{origin_y+26}" text-anchor="middle" font-size="13" font-family="Menlo, Monaco, monospace" fill="#334155">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def line_chart_svg(
    title: str,
    series: Mapping[str, List[float]],
    x_labels: List[str],
    *,
    subtitle: str = "",
    width: int = 960,
    height: int = 520,
    formatter: Callable[[float], str] = _format_float,
) -> str:
    if not series:
        raise ValueError("line chart requires series")
    max_value = max((max(values) for values in series.values() if values), default=1.0) or 1.0
    left, right, top, bottom = 90, 40, 110, 100
    chart_width = width - left - right
    chart_height = height - top - bottom
    origin_y = top + chart_height
    colors = ["#2463EB", "#DC2626", "#059669", "#7C3AED", "#EA580C"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        f'<text x="{left}" y="44" font-size="28" font-family="Menlo, Monaco, monospace" fill="#0F172A">{title}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{left}" y="72" font-size="14" font-family="Menlo, Monaco, monospace" fill="#475569">{subtitle}</text>'
        )
    for i in range(5):
        tick_value = max_value * i / 4
        y = origin_y - chart_height * i / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-size="12" font-family="Menlo, Monaco, monospace" fill="#64748B">{formatter(tick_value)}</text>'
        )

    xs = [left + chart_width / 2] if len(x_labels) == 1 else [left + i * (chart_width / (len(x_labels) - 1)) for i in range(len(x_labels))]
    for x, label in zip(xs, x_labels):
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{origin_y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{x}" y="{origin_y+26}" text-anchor="middle" font-size="13" font-family="Menlo, Monaco, monospace" fill="#334155">{label}</text>'
        )

    legend_x = left
    legend_y = height - 28
    for color, (name, values) in zip(colors, series.items()):
        points = []
        for x, value in zip(xs, values):
            y = origin_y - (0 if max_value == 0 else chart_height * (value / max_value))
            points.append((x, y, value))
        polyline = " ".join(f"{x},{y}" for x, y, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
            parts.append(
                f'<text x="{x}" y="{y-10}" text-anchor="middle" font-size="12" font-family="Menlo, Monaco, monospace" fill="{color}">{formatter(value)}</text>'
            )
        parts.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x+22}" y="{legend_y+2}" font-size="13" font-family="Menlo, Monaco, monospace" fill="#334155">{name}</text>'
        )
        legend_x += 180
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_chart_svg(
    title: str,
    points: List[dict],
    *,
    subtitle: str = "",
    width: int = 960,
    height: int = 520,
) -> str:
    if not points:
        raise ValueError("scatter chart requires at least one point")
    max_x = max(point["x"] for point in points) or 1.0
    max_y = max(point["y"] for point in points) or 1.0
    left, right, top, bottom = 90, 40, 110, 100
    chart_width = width - left - right
    chart_height = height - top - bottom
    origin_y = top + chart_height
    colors = {
        "standard": "#2463EB",
        "preset_3p5_qjl": "#059669",
        "preset_2p5_qjl": "#EA580C",
        "core_4bit": "#7C3AED",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        f'<text x="{left}" y="44" font-size="28" font-family="Menlo, Monaco, monospace" fill="#0F172A">{title}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{left}" y="72" font-size="14" font-family="Menlo, Monaco, monospace" fill="#475569">{subtitle}</text>'
        )
    for i in range(5):
        x = left + chart_width * i / 4
        y = origin_y - chart_height * i / 4
        x_value = max_x * i / 4
        y_value = max_y * i / 4
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{origin_y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#E2E8F0" stroke-width="1"/>')
        parts.append(
            f'<text x="{x}" y="{origin_y+26}" text-anchor="middle" font-size="12" font-family="Menlo, Monaco, monospace" fill="#64748B">{int(x_value):,}</text>'
        )
        parts.append(
            f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-size="12" font-family="Menlo, Monaco, monospace" fill="#64748B">{y_value:.3f}</text>'
        )

    for point in points:
        x = left + (0 if max_x == 0 else chart_width * (point["x"] / max_x))
        y = origin_y - (0 if max_y == 0 else chart_height * (point["y"] / max_y))
        color = colors.get(point["mode_slug"], "#334155")
        parts.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{color}" opacity="0.9"/>')
        parts.append(
            f'<text x="{x+10}" y="{y-10}" font-size="12" font-family="Menlo, Monaco, monospace" fill="#0F172A">{point["label"]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
