# 本文件实现布图结果的几何后处理，
# 在不改变模块尺寸和旋转状态的前提下，将合法矩形尽量向左、向下压紧。
"""Post-processing helpers for compacting legal floorplans."""


from typing import Dict, List, Tuple

from .models import FloorplanResult, Module


def _effective_size(module: Module, rotated: bool) -> Tuple[float, float]:
    """根据旋转状态得到模块实际宽高。"""
    if rotated and module.is_hard:
        return module.height, module.width
    return module.width, module.height


def _interval_overlap(a0: float, a1: float, b0: float, b1: float,
                      eps: float = 1e-9) -> bool:
    """判断两个一维开区间是否相交；边界相贴不算相交。"""
    return a0 < b1 - eps and b0 < a1 - eps


def compact_floorplan(result: FloorplanResult, modules: List[Module],
                      rounds: int = 20) -> FloorplanResult:
    """
    对合法布图做左推/下推压缩。

    中文说明：这是最终几何紧凑化后处理，不改变 B*-Tree 搜索得到的
    模块集合、旋转状态和相互不重叠约束，只是在保持合法的前提下
    消除能直接推掉的横向/纵向空隙。
    """
    module_map = {module.name: module for module in modules}
    positions: Dict[str, List[float]] = {
        name: [x, y, rotated]
        for name, (x, y, rotated) in result.module_positions.items()
    }

    for _ in range(rounds):
        changed = False

        for name in sorted(positions, key=lambda item: positions[item][0]):
            module = module_map.get(name)
            if module is None:
                continue
            x, y, rotated = positions[name]
            w, h = _effective_size(module, rotated)
            new_x = 0.0

            for other_name, (ox, oy, other_rotated) in positions.items():
                if other_name == name:
                    continue
                other_module = module_map.get(other_name)
                if other_module is None:
                    continue
                ow, oh = _effective_size(other_module, other_rotated)
                if (
                    ox + ow <= x + 1e-9
                    and _interval_overlap(y, y + h, oy, oy + oh)
                ):
                    new_x = max(new_x, ox + ow)

            if new_x < x - 1e-9:
                positions[name][0] = new_x
                changed = True

        for name in sorted(positions, key=lambda item: positions[item][1]):
            module = module_map.get(name)
            if module is None:
                continue
            x, y, rotated = positions[name]
            w, h = _effective_size(module, rotated)
            new_y = 0.0

            for other_name, (ox, oy, other_rotated) in positions.items():
                if other_name == name:
                    continue
                other_module = module_map.get(other_name)
                if other_module is None:
                    continue
                ow, oh = _effective_size(other_module, other_rotated)
                if (
                    oy + oh <= y + 1e-9
                    and _interval_overlap(x, x + w, ox, ox + ow)
                ):
                    new_y = max(new_y, oy + oh)

            if new_y < y - 1e-9:
                positions[name][1] = new_y
                changed = True

        if not changed:
            break

    max_x = 0.0
    max_y = 0.0
    compacted_positions = {}
    total_block_area = 0.0
    for module in modules:
        total_block_area += module.area
        if module.name not in positions:
            continue
        x, y, rotated = positions[module.name]
        w, h = _effective_size(module, rotated)
        compacted_positions[module.name] = (x, y, rotated)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    area = max_x * max_y
    aspect_ratio = max(max_x, max_y) / max(1e-9, min(max_x, max_y))
    dead_space_ratio = (area - total_block_area) / max(total_block_area, 1e-9)

    return FloorplanResult(
        module_positions=compacted_positions,
        outline_width=max_x,
        outline_height=max_y,
        area=area,
        aspect_ratio=aspect_ratio,
        total_hpwl=result.total_hpwl,
        dead_space_ratio=dead_space_ratio,
    )
