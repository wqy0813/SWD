# 本文件提供矩形布图结果的严格几何校验工具，
# 用于检查模块是否缺失、是否越界、以及任意两个模块是否重叠。
"""Validation helpers for VLSI floorplanning results."""


from typing import List, Tuple

from .models import FloorplanResult, Module


def _effective_size(module: Module, rotated: bool) -> Tuple[float, float]:
    """根据结果中的旋转标记计算模块实际宽高。"""
    if rotated and module.is_hard:
        return module.height, module.width
    return module.width, module.height


def _overlap(a_x: float, a_y: float, a_w: float, a_h: float,
             b_x: float, b_y: float, b_w: float, b_h: float,
             eps: float) -> bool:
    """判断两个矩形内部是否相交；边界相贴不算重叠。"""
    return (
        a_x < b_x + b_w - eps
        and b_x < a_x + a_w - eps
        and a_y < b_y + b_h - eps
        and b_y < a_y + a_h - eps
    )


def validate_floorplan(result: FloorplanResult, modules: List[Module],
                       eps: float = 1e-6) -> Tuple[bool, List[str]]:
    """严格校验布图结果是否合法。"""
    issues = []
    module_map = {module.name: module for module in modules}

    missing = sorted(set(module_map) - set(result.module_positions))
    if missing:
        issues.append(f"缺失模块: {', '.join(missing[:8])}")

    rects = []
    for name, (x, y, rotated) in result.module_positions.items():
        module = module_map.get(name)
        if module is None:
            issues.append(f"未知模块: {name}")
            continue

        w, h = _effective_size(module, rotated)
        if x < -eps or y < -eps:
            issues.append(f"{name} 坐标为负: ({x:.3f}, {y:.3f})")
        if x + w > result.outline_width + eps:
            issues.append(
                f"{name} 越过右边界: {x + w:.3f} > {result.outline_width:.3f}"
            )
        if y + h > result.outline_height + eps:
            issues.append(
                f"{name} 越过上边界: {y + h:.3f} > {result.outline_height:.3f}"
            )
        rects.append((name, x, y, w, h))

    for i in range(len(rects)):
        name_i, xi, yi, wi, hi = rects[i]
        for j in range(i + 1, len(rects)):
            name_j, xj, yj, wj, hj = rects[j]
            if _overlap(xi, yi, wi, hi, xj, yj, wj, hj, eps):
                issues.append(f"{name_i} 与 {name_j} 重叠")

    return len(issues) == 0, issues
