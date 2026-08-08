# 本文件定义 VLSI 布图规划项目的核心数据结构，包括模块、子矩形、
# 连线网络和求解结果对象，供解析器、优化器和可视化模块共同使用。
"""VLSI floorplanning solver modules split from the original spr backup."""


from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Module:
    """Represents a functional block (HardBlock or Terminal)."""
    name: str
    module_type: str  # 'block' or 'terminal'
    width: float = 0.0
    height: float = 0.0
    x: float = 0.0       # final x position (bottom-left)
    y: float = 0.0       # final y position (bottom-left)
    rotated: bool = False  # whether the module is rotated 90 degrees
    is_hard: bool = True   # True for block, False for terminal (fixed)
    # For non-rectangular modules (Problem 4)
    shape_type: str = 'rect'  # 'rect', 'L', 'T'
    sub_blocks: List['SubBlock'] = field(default_factory=list)

    @property
    def w(self):
        """Effective width after rotation."""
        return self.height if self.rotated and self.is_hard else self.width

    @property
    def h(self):
        """Effective height after rotation."""
        return self.width if self.rotated and self.is_hard else self.height

    @property
    def area(self):
        return self.width * self.height



@dataclass
class SubBlock:
    """Sub-block for non-rectangular modules (L-shape, T-shape)."""
    rel_x: float  # relative x offset from module origin
    rel_y: float  # relative y offset from module origin
    width: float
    height: float



@dataclass
class Net:
    """Represents a connection network."""
    name: str = ""
    pins: List[str] = field(default_factory=list)  # list of module names
    weight: float = 1.0



@dataclass
class FloorplanResult:
    """Stores the result of a floorplanning run."""
    module_positions: Dict[str, Tuple[float, float, bool]]  # name -> (x, y, rotated)
    outline_width: float
    outline_height: float
    area: float
    aspect_ratio: float
    total_hpwl: float = 0.0
    dead_space_ratio: float = 0.0
    feasible: bool = True
    verify_info: str = ""


@dataclass
class Problem1Config:
    """问题一专用参数：无固定轮廓，优先压缩面积，再优化长宽比。"""
    seeds: Tuple[int, ...] = (42, 142, 242, 342, 442)
    initial_tree_modes: Tuple[str, ...] = ("random_balanced",)
    area_tie_epsilon: float = 1e-6
    alpha: float = 0.5
    beta: float = 0.0
    target_aspect_ratio: float = 1.0
    t_initial: float = 10.0
    t_final: float = 0.001
    cooling_rate: float = 0.95
    min_total_iter: int = 24000
    iter_multiplier: int = 240
    min_iter_per_temp: int = 30
    stage1_shape_weight: float = 0.0
    stage2_area_budget_factor: float = 1.0
    stage2_area_penalty: float = 100.0
    stage2_min_total_iter: int = 6000
    stage2_iter_multiplier: int = 60
    stage1_aspect_ratio_limit: float = 1.0e9
    stage1_aspect_ratio_penalty: float = 0.0


@dataclass
class Problem2Config:
    """问题二专用参数：固定正方形轮廓，主要优化 HPWL 并惩罚越界。"""
    dead_space_ratio: float = 0.15
    seed: int = 123
    seeds: Tuple[int, ...] = (123, 223, 323, 423, 523)
    alpha: float = 0.2
    beta: float = 0.6
    target_aspect_ratio: float = 1.0
    t_initial: float = 5000.0
    t_final: float = 0.01
    cooling_rate: float = 0.95
    min_total_iter: int = 5000
    iter_multiplier: int = 80
    min_iter_per_temp: int = 40
    outline_penalty: float = 100000.0
    feasibility_tolerance: float = 1e-6


@dataclass
class Problem3Config:
    """问题三专用参数：二分搜索最小可行死区比例。"""
    low: float = 0.0
    high: float = 0.5
    precision: float = 0.001
    max_binary_iter: int = 30
    trials_per_ratio: int = 3
    t_initial: float = 3000.0
    t_final: float = 0.1
    cooling_rate: float = 0.93
    min_total_iter: int = 3000
    iter_multiplier: int = 50
    min_iter_per_temp: int = 30
    outline_penalty: float = 50000.0


@dataclass
class Problem4Config:
    """问题四专用参数：异形模块拆分与小规模非矩形布局搜索。"""
    seed: int = 42
    t_initial: float = 5000.0
    t_final: float = 0.01
    cooling_rate: float = 0.95
    min_total_iter: int = 2000
    iter_multiplier: int = 80
    min_iter_per_temp: int = 30


# ============================================================
# FILE PARSER
# ============================================================
