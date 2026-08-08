# 本文件实现布图结果的评价指标，目前主要计算 HPWL 半周长线长，
# 为问题 2 和问题 3 的连线优化目标提供统一接口。
"""VLSI floorplanning solver modules split from the original spr backup."""


from typing import Dict, List, Tuple

from .models import Module, Net

def compute_hpwl(nets: List[Net],
                 module_positions: Dict[str, Tuple[float, float]],
                 modules: List[Module],
                 terminal_positions: Dict[str, Tuple[float, float]]) -> float:
    """
    Compute total Half-Perimeter Wirelength (HPWL) for all nets.

    For each net, HPWL = (max_x - min_x) + (max_y - min_y)
    where min/max are taken over all pins in the net.
    """
    total_hpwl = 0.0

    # Build pin position lookup
    pin_positions = {}
    for mod in modules:
        if mod.name in module_positions:
            pos = module_positions[mod.name]
            x, y = pos[0], pos[1]
            rotated = pos[2] if len(pos) >= 3 else mod.rotated
            width = mod.height if rotated and mod.is_hard else mod.width
            height = mod.width if rotated and mod.is_hard else mod.height
            # Pin is at module center
            pin_positions[mod.name] = (x + width / 2.0, y + height / 2.0)

    for name, (tx, ty) in terminal_positions.items():
        pin_positions[name] = (tx, ty)

    for net in nets:
        if not net.pins:
            continue
        xs = []
        ys = []
        for pin_name in net.pins:
            if pin_name in pin_positions:
                px, py = pin_positions[pin_name]
                xs.append(px)
                ys.append(py)

        if xs:
            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            total_hpwl += hpwl * net.weight

    return total_hpwl


# ============================================================
# SIMULATED ANNEALING ENGINE
# ============================================================
