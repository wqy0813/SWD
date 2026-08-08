# 本文件负责将求解得到的模块坐标绘制成布局图，
# 可选择叠加连线网络，用于论文结果展示和算法调试。
"""VLSI floorplanning solver modules split from the original spr backup."""


from typing import Dict, List, Tuple

from .models import FloorplanResult, Module, Net

def visualize_floorplan(result: FloorplanResult,
                        modules: List[Module],
                        title: str = "Floorplan",
                        save_path: str = None,
                        show_nets: bool = False,
                        nets: List[Net] = None,
                        terminal_positions: Dict[str, Tuple[float, float]] = None):
    """Generate a visualization of the floorplan."""
    if not _HAS_MATPLOTLIB:
        if save_path:
            print(f"[SKIP] Cannot save {save_path} — matplotlib not installed")
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Color map for different modules
    colors = plt.cm.Set3(range(len(modules)))

    # Draw modules
    for i, mod in enumerate(modules):
        if mod.name in result.module_positions:
            x, y, rotated = result.module_positions[mod.name]
            w, h = mod.w, mod.h
            color = colors[i % len(colors)]

            if mod.shape_type == 'rect':
                rect = Rectangle((x, y), w, h, linewidth=1.5,
                                 edgecolor='black', facecolor=color, alpha=0.7)
                ax.add_patch(rect)
            elif mod.sub_blocks:
                # Draw each sub-block
                for sb in mod.sub_blocks:
                    rect = Rectangle((x + sb.rel_x, y + sb.rel_y),
                                     sb.width, sb.height, linewidth=1.0,
                                     edgecolor='black', facecolor=color, alpha=0.7)
                    ax.add_patch(rect)

            # Label
            ax.text(x + w / 2.0, y + h / 2.0, mod.name,
                    ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw outline
    outline = Rectangle((0, 0), result.outline_width, result.outline_height,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--')
    ax.add_patch(outline)

    # Draw terminal positions if available
    if terminal_positions:
        for name, (tx, ty) in terminal_positions.items():
            ax.plot(tx, ty, 'r*', markersize=8)
            ax.text(tx + 1, ty + 1, name, fontsize=5, color='red')

    # Draw nets if requested
    if show_nets and nets and terminal_positions:
        pin_positions = {}
        for mod in modules:
            if mod.name in result.module_positions:
                x, y, _ = result.module_positions[mod.name]
                pin_positions[mod.name] = (x + mod.w / 2.0, y + mod.h / 2.0)
        for name, (tx, ty) in terminal_positions.items():
            pin_positions[name] = (tx, ty)

        for net in nets[:30]:  # Limit to first 30 nets for clarity
            if len(net.pins) >= 2:
                for k in range(len(net.pins) - 1):
                    if net.pins[k] in pin_positions and net.pins[k + 1] in pin_positions:
                        x1, y1 = pin_positions[net.pins[k]]
                        x2, y2 = pin_positions[net.pins[k + 1]]
                        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=0.3, alpha=0.5)

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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


# ============================================================
# MAIN ENTRY POINT
# ============================================================
