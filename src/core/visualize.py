# 本文件负责将求解得到的模块坐标绘制成布局图，
# 可选择叠加连线网络，用于论文结果展示和算法调试。
"""VLSI floorplanning solver modules split from the original spr backup."""


import os
from typing import Dict, List, Tuple

from .models import FloorplanResult, Module, Net

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _effective_rect_size(mod: Module, rotated: bool) -> Tuple[float, float]:
    """根据求解结果中的旋转状态计算矩形模块的实际宽高。"""
    if rotated and mod.is_hard:
        return mod.height, mod.width
    return mod.width, mod.height


def _build_color_palette(count: int):
    """生成低饱和度马卡龙色系，兼顾大量模块时的区分度。"""
    macaron_hex = [
        "#F4B6C2", "#F6D6AD", "#F7E7A6", "#C7E5B4", "#A9D8C8",
        "#AFCDE7", "#C8BFE7", "#E7BEDA", "#F2C6B4", "#D6E6A9",
        "#B7E1D4", "#B8D8E8", "#D4C5E8", "#EAC7D8", "#F1D0A8",
        "#E8E0A8", "#C9DEB9", "#B5D7CE", "#B9CFE3", "#D9C3D8",
        "#E9BBB5", "#F3D7BF", "#DDE7B7", "#BBDDC1", "#B8D6DA",
        "#C6CCE6", "#DDC4E2", "#EBC4C4", "#EED3A7", "#D7E3C3",
    ]
    palette = [mcolors.to_rgb(color) for color in macaron_hex]

    if count > len(palette):
        # 中文说明：模块数量超过基础色板时，用黄金比例分散 hue，
        # 但保持低饱和和高明度，避免回到刺眼高饱和色。
        for i in range(count - len(palette)):
            hue = (i * 0.61803398875) % 1.0
            saturation = 0.18 + 0.08 * ((i % 3) / 2.0)
            value = 0.92 + 0.04 * (i % 2)
            palette.append(mcolors.hsv_to_rgb((hue, saturation, value)))

    return palette[:max(count, 1)]


def _should_draw_module_label(name: str, width: float, height: float,
                              result: FloorplanResult, module_count: int) -> bool:
    """自动隐藏过密标签，只给视觉上有空间的模块标名字。"""
    if module_count <= 80:
        return True

    outline_area = max(result.outline_width * result.outline_height, 1e-9)
    area_ratio = (width * height) / outline_area
    min_outline = max(min(result.outline_width, result.outline_height), 1e-9)
    min_side_ratio = min(width, height) / min_outline

    if module_count <= 150:
        return area_ratio >= 0.0025 and min_side_ratio >= 0.018
    if module_count <= 250:
        return area_ratio >= 0.005 and min_side_ratio >= 0.026
    return area_ratio >= 0.008 and min_side_ratio >= 0.034


def visualize_floorplan(result: FloorplanResult,
                        modules: List[Module],
                        title: str = "Floorplan",
                        save_path: str = None,
                        show_nets: bool = False,
                        nets: List[Net] = None,
                        terminal_positions: Dict[str, Tuple[float, float]] = None,
                        label_all_modules: bool = True,
                        label_terminals: bool = False):
    """Generate a visualization of the floorplan."""
    if not _HAS_MATPLOTLIB:
        if save_path:
            print(f"[SKIP] Cannot save {save_path} - matplotlib not installed")
        return
    module_count = len(modules)
    fig_size = 12 if module_count <= 120 else (16 if module_count <= 240 else 20)
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

    # 中文说明：使用离散调色板循环，不再直接把整数传给 colormap。
    colors = _build_color_palette(len(modules))
    module_count = len(modules)
    label_fontsize = 8 if module_count <= 80 else (5.5 if module_count <= 200 else 4.2)
    edge_width = 1.2 if module_count <= 100 else (0.8 if module_count <= 220 else 0.55)
    edge_color = "#565656" if module_count <= 150 else "#6E6E6E"

    # Draw modules
    for i, mod in enumerate(modules):
        if mod.name in result.module_positions:
            x, y, rotated = result.module_positions[mod.name]
            w, h = _effective_rect_size(mod, rotated)
            color = colors[i % len(colors)]

            if mod.shape_type == 'rect':
                rect = Rectangle((x, y), w, h, linewidth=edge_width,
                                 edgecolor=edge_color, facecolor=color, alpha=0.82)
                ax.add_patch(rect)
            elif mod.sub_blocks:
                # Draw each sub-block
                for sb in mod.sub_blocks:
                    rect = Rectangle((x + sb.rel_x, y + sb.rel_y),
                                     sb.width, sb.height, linewidth=edge_width,
                                     edgecolor=edge_color, facecolor=color, alpha=0.82)
                    ax.add_patch(rect)

            # Label
            if label_all_modules or _should_draw_module_label(mod.name, w, h, result, module_count):
                ax.text(x + w / 2.0, y + h / 2.0, mod.name,
                        ha='center', va='center', fontsize=label_fontsize,
                        color="#333333")

    # Draw outline
    outline = Rectangle((0, 0), result.outline_width, result.outline_height,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--')
    ax.add_patch(outline)

    # Draw terminal positions if available
    if terminal_positions:
        terminal_count = len(terminal_positions)
        terminal_marker_size = 8 if terminal_count <= 80 else (4 if terminal_count <= 250 else 2.5)
        for name, (tx, ty) in terminal_positions.items():
            ax.plot(tx, ty, 'r*', markersize=terminal_marker_size)
            if label_terminals or terminal_count <= 60:
                ax.text(tx + 1, ty + 1, name, fontsize=5, color='red')

    # Draw nets if requested
    if show_nets and nets and terminal_positions:
        pin_positions = {}
        for mod in modules:
            if mod.name in result.module_positions:
                x, y, rotated = result.module_positions[mod.name]
                w, h = _effective_rect_size(mod, rotated)
                pin_positions[mod.name] = (x + w / 2.0, y + h / 2.0)
        for name, (tx, ty) in terminal_positions.items():
            pin_positions[name] = (tx, ty)

        for net in nets[:30]:  # Limit to first 30 nets for clarity
            points = [pin_positions[pin] for pin in net.pins if pin in pin_positions]
            if len(points) < 2:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                ax.plot([x1, x2], [y1, y1], color="#1D4ED8",
                        linewidth=0.45, alpha=0.55)
                ax.plot([x2, x2], [y1, y2], color="#1D4ED8",
                        linewidth=0.45, alpha=0.55)
            elif abs(max_x - min_x) < 1e-9 or abs(max_y - min_y) < 1e-9:
                ax.plot([min_x, max_x], [min_y, max_y], color="#1D4ED8",
                        linewidth=0.45, alpha=0.45)
            else:
                hpwl_box = Rectangle(
                    (min_x, min_y),
                    max_x - min_x,
                    max_y - min_y,
                    linewidth=0.45,
                    edgecolor="#1D4ED8",
                    facecolor="none",
                    linestyle=":",
                    alpha=0.45,
                )
                ax.add_patch(hpwl_box)

    ax.set_xlim(-10, result.outline_width * 1.1)
    ax.set_ylim(-10, result.outline_height * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    # Add info text
    info_text = (f"Area: {result.area:.2f}\n"
                 f"Outline: {result.outline_width:.2f} x {result.outline_height:.2f}\n"
                 f"Aspect Ratio: {result.aspect_ratio:.4f}")
    if result.total_hpwl > 0:
        info_text += f"\nHPWL: {result.total_hpwl:.2f}"
    if result.dead_space_ratio > 0:
        info_text += f"\nDSR: {result.dead_space_ratio:.4f}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    try:
        plt.tight_layout()
    except Exception:
        pass

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


# ============================================================
# MAIN ENTRY POINT
# ============================================================
