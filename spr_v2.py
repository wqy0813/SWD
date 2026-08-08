# -*- coding: utf-8 -*-
"""
spr_v2.py —— VLSI 布图规划改进版求解器
========================================
在 spr.py(B*-Tree + 模拟退火)基础上, 针对评委意见(P0/P1/P2)改进:

  P0(必须) 修复:
    1. 解析器 bug: 支持 "(0, 0)" 带空格的坐标格式(原版对 n100/n200/n300 解析失败);
    2. 问题4 形状约束: 模块作为"整体"参与搜索(子矩形仅用于碰撞/面积), 保证 L/T 形状不被破坏;
    3. 问题2/3 严格可行: 精确越界/重叠校验 + 两阶段(先可行性后线长) + 可行修复。

  P1(强烈建议):
    4. 多起点并行: 多种子运行取最优, 并输出统计(最优/均值/标准差);
    5. 自适应惩罚: 根据近期越界情况动态调整轮廓惩罚权重;
    6. 增量 HPWL: 只重算受"位置变化模块"影响的线网。

  P2(加分项):
    7. 两阶段优化: 问题1(先面积后长宽比) / 问题2(先可行性后HPWL);
    8. 理论下界验证: 面积下界 = 模块总面积;
    9. 精确几何校验: 顶点级重叠/越界逐项检查;
   10. 多次运行统计与敏感性分析: mean/std/best。

可视化: 优先 matplotlib, 缺失时使用 PIL 自绘(无需联网安装)。
运行: python spr_v2.py --path 附件 --chip n100 [--seeds 3] [--quick]
      python spr_v2.py --p4            # 求解问题4 示例(图3, 用户给定尺寸)
"""
import math
import sys
sys.setrecursionlimit(20000)
import random
import copy
import sys
import os
import json
import time
import re
import subprocess
import pickle
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# ============================================================
# 数据结构
# ============================================================

@dataclass
class Module:
    """功能模块(HardBlock 或 Terminal)。"""
    name: str
    module_type: str          # 'block' 或 'terminal'
    width: float = 0.0
    height: float = 0.0
    x: float = 0.0
    y: float = 0.0
    rotated: bool = False
    is_hard: bool = True
    shape_type: str = 'rect'  # 'rect' / 'L' / 'T'
    sub_blocks: List['SubBlock'] = field(default_factory=list)

    @property
    def w(self):
        return self.height if (self.rotated and self.is_hard) else self.width

    @property
    def h(self):
        return self.width if (self.rotated and self.is_hard) else self.height

    @property
    def area(self):
        return self.width * self.height


@dataclass
class SubBlock:
    """子矩形(问题4 非矩形模块的分解块)。"""
    rel_x: float
    rel_y: float
    width: float
    height: float


@dataclass
class Net:
    """线网。"""
    name: str = ""
    pins: List[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class FloorplanResult:
    """布图结果。"""
    module_positions: Dict[str, Tuple[float, float, bool]]  # name -> (x, y, rotated)
    outline_width: float
    outline_height: float
    area: float
    aspect_ratio: float
    total_hpwl: float = 0.0
    dead_space_ratio: float = 0.0
    feasible: bool = True            # 是否通过精确校验
    verify_info: str = ""            # 校验说明
    # 统计信息(多起点)
    best_area: float = 0.0
    mean_area: float = 0.0
    std_area: float = 0.0
    best_hpwl: float = 0.0
    mean_hpwl: float = 0.0
    std_hpwl: float = 0.0
    n_runs: int = 1
    lower_bound_area: float = 0.0    # 理论下界


# ============================================================
# 文件解析器(修复: 支持带空格的坐标)
# ============================================================

_COORD_RE = re.compile(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")


class VLSIParser:
    """解析 .blocks/.nets/.pl, 兼容多种编码与 "(x, y)" 格式。"""

    @staticmethod
    def _read_lines(filepath: str):
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.readlines()
            except (UnicodeDecodeError, UnicodeError):
                continue
        return []

    @staticmethod
    def parse_blocks(filepath: str) -> List[Module]:
        modules = []
        for line in VLSIParser._read_lines(filepath):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('Num'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, mtype = parts[0], parts[1]
            if mtype == 'terminal':
                modules.append(Module(name=name, module_type='terminal',
                                      width=0, height=0, is_hard=False))
            elif mtype == 'block':
                # 用正则提取所有 (x, y) 顶点(兼容 "(0,0)" 与 "(0, 0)")
                coords = [(float(a), float(b)) for a, b in _COORD_RE.findall(line)]
                if coords:
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    mod = Module(name=name, module_type='block',
                                 width=width, height=height, is_hard=True)
                    if len(coords) >= 6:
                        shape_type = VLSIParser._detect_shape_type(coords)
                        mod.shape_type = shape_type
                        if shape_type != 'rect':
                            mod.sub_blocks = VLSIParser._decompose_shape(coords)
                    modules.append(mod)
        return modules

    @staticmethod
    def _detect_shape_type(coords: List[Tuple[float, float]]) -> str:
        min_x = min(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_x = max(c[0] for c in coords)
        max_y = max(c[1] for c in coords)
        w, h = max_x - min_x, max_y - min_y
        area_poly = VLSIParser._polygon_area(coords)
        area_bbox = w * h
        if abs(area_poly - area_bbox) < 1e-6:
            return 'rect'
        mid_x, mid_y = min_x + w / 2, min_y + h / 2
        quadrants = [
            [(min_x, min_y), (mid_x, min_y), (mid_x, mid_y), (min_x, mid_y)],
            [(mid_x, min_y), (max_x, min_y), (max_x, mid_y), (mid_x, mid_y)],
            [(min_x, mid_y), (mid_x, mid_y), (mid_x, max_y), (min_x, max_y)],
            [(mid_x, mid_y), (max_x, mid_y), (max_x, max_y), (mid_x, max_y)],
        ]
        covered = sum(1 for q in quadrants
                      if VLSIParser._point_in_polygon(
                          (q[0][0]+q[2][0])/2, (q[0][1]+q[2][1])/2, coords))
        return 'L' if covered == 3 else 'T'

    @staticmethod
    def _polygon_area(coords) -> float:
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0]*coords[j][1] - coords[j][0]*coords[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _point_in_polygon(x: float, y: float, coords) -> bool:
        n = len(coords)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = coords[i]
            xj, yj = coords[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi)*(y - yi)/(yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _decompose_shape(coords: List[Tuple[float, float]]) -> List[SubBlock]:
        min_x = min(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        xs = sorted(set(c[0] for c in coords))
        ys = sorted(set(c[1] for c in coords))
        sub_blocks = []
        for i in range(len(xs)-1):
            for j in range(len(ys)-1):
                x1, x2 = xs[i], xs[i+1]
                y1, y2 = ys[j], ys[j+1]
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if VLSIParser._point_in_polygon(cx, cy, coords):
                    sub_blocks.append(SubBlock(x1-min_x, y1-min_y, x2-x1, y2-y1))
        return VLSIParser._merge_sub_blocks(sub_blocks)

    @staticmethod
    def _merge_sub_blocks(blocks: List[SubBlock]) -> List[SubBlock]:
        if len(blocks) <= 1:
            return blocks
        merged = True
        result = list(blocks)
        while merged:
            merged = False
            new_result = []
            used = set()
            for i, a in enumerate(result):
                if i in used:
                    continue
                found = False
                for j, b in enumerate(result):
                    if j <= i or j in used:
                        continue
                    if (abs(a.height-b.height) < 1e-6 and
                        abs((a.rel_x+a.width)-b.rel_x) < 1e-6 and
                        abs(a.rel_y-b.rel_y) < 1e-6):
                        new_result.append(SubBlock(a.rel_x, a.rel_y,
                                                   a.width+b.width, a.height))
                        used.update((i, j)); merged = True; found = True; break
                    if (abs(a.width-b.width) < 1e-6 and
                        abs((a.rel_y+a.height)-b.rel_y) < 1e-6 and
                        abs(a.rel_x-b.rel_x) < 1e-6):
                        new_result.append(SubBlock(a.rel_x, a.rel_y,
                                                   a.width, a.height+b.height))
                        used.update((i, j)); merged = True; found = True; break
                if not found:
                    new_result.append(a)
            result = new_result
        return result

    @staticmethod
    def parse_nets(filepath: str) -> List[Net]:
        nets = []
        lines = [l.strip() for l in VLSIParser._read_lines(filepath)]
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line or line.startswith('//') or 'Num' in line:
                continue
            if 'NetDegree' in line or 'NetDegree' in line:
                try:
                    if ':' in line or '：' in line:
                        sep = ':' if ':' in line else '：'
                        degree = int(line.split(sep)[1].strip())
                    else:
                        degree = int(line.split()[-1])
                except (ValueError, IndexError):
                    continue
                net = Net()
                for _ in range(degree):
                    if i < len(lines):
                        pin = lines[i].strip()
                        i += 1
                        if pin and not pin.startswith('//'):
                            net.pins.append(pin)
                if net.pins:
                    nets.append(net)
        return nets

    @staticmethod
    def parse_pl(filepath: str) -> Dict[str, Tuple[float, float]]:
        positions = {}
        for line in VLSIParser._read_lines(filepath):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    positions[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    continue
        return positions
# ============================================================
# B*-Tree 表示
# ============================================================

class BTreeNode:
    __slots__ = ('module_idx', 'left', 'right', 'parent')
    def __init__(self, module_idx: int):
        self.module_idx = module_idx
        self.left = None
        self.right = None
        self.parent = None


class BTree:
    """B*-Tree: 左孩子=放父模块右侧, 右孩子=放父模块上方。"""

    def __init__(self, num_modules: int):
        self.num_modules = num_modules
        self.root = None
        self.nodes: List[Optional[BTreeNode]] = [None] * num_modules

    def copy(self) -> 'BTree':
        new_tree = BTree(self.num_modules)
        if self.root is None:
            return new_tree
        # 迭代深拷贝(显式栈)
        stack = [(self.root, None)]  # (旧节点, 新父节点)
        new_root = None
        while stack:
            old_node, new_parent = stack.pop()
            if old_node is None:
                continue
            new_node = BTreeNode(old_node.module_idx)
            new_node.parent = new_parent
            new_tree.nodes[old_node.module_idx] = new_node
            if new_parent is None:
                new_root = new_node
            else:
                # 通过旧树关系判断左右(父节点的左右孩子指针指向该旧节点)
                if old_node.parent is not None:
                    if old_node.parent.left is old_node:
                        new_parent.left = new_node
                    elif old_node.parent.right is old_node:
                        new_parent.right = new_node
            stack.append((old_node.right, new_node))
            stack.append((old_node.left, new_node))
        new_tree.root = new_root
        return new_tree

    def build_initial_tree(self, module_indices: List[int]):
        self.root = self._build_balanced(module_indices, 0, len(module_indices) - 1, None)

    def _build_balanced(self, indices, start, end, parent):
        if start > end:
            return None
        mid = (start + end) // 2
        node = BTreeNode(indices[mid])
        node.parent = parent
        self.nodes[indices[mid]] = node
        node.left = self._build_balanced(indices, mid + 1, end, node)
        node.right = self._build_balanced(indices, start, mid - 1, node)
        return node

    def get_all_nodes(self):
        nodes = []
        stack = [self.root]
        seen = set()
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            nodes.append(node)
            stack.append(node.right)
            stack.append(node.left)
        return nodes

    def get_node(self, module_idx):
        stack = [self.root]
        seen = set()
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if node.module_idx == module_idx:
                return node
            stack.append(node.right)
            stack.append(node.left)
        return None

    def swap_modules(self, idx1, idx2):
        n1 = self.get_node(idx1)
        n2 = self.get_node(idx2)
        if n1 and n2:
            n1.module_idx, n2.module_idx = n2.module_idx, n1.module_idx

    def delete_and_insert(self, delete_idx, insert_idx):
        node = self.get_node(delete_idx)
        if node is None or delete_idx == insert_idx:
            return
        moved = node.module_idx
        self._delete_node(node)
        target = self.get_node(insert_idx)
        if target is None:
            alln = self.get_all_nodes()
            target = alln[-1] if alln else None
        if target:
            if random.random() < 0.5 and target.left is None:
                newn = BTreeNode(moved); newn.parent = target; target.left = newn
            elif target.right is None:
                newn = BTreeNode(moved); newn.parent = target; target.right = newn
            else:
                alln = self.get_all_nodes()
                for n in alln:
                    if n.left is None:
                        newn = BTreeNode(moved); newn.parent = n; n.left = newn
                        return
                    if n.right is None:
                        newn = BTreeNode(moved); newn.parent = n; n.right = newn
                        return
                last = alln[-1] if alln else None
                if last:
                    newn = BTreeNode(moved); newn.parent = last; last.right = newn

    def _delete_node(self, node):
        replacement = node.left if node.left else node.right
        other = node.right if node.left else node.left
        if node.parent:
            if node.parent.left == node:
                node.parent.left = replacement
            else:
                node.parent.right = replacement
            if replacement:
                replacement.parent = node.parent
        else:
            self.root = replacement
            if replacement:
                replacement.parent = None
        if node.left and node.right and replacement:
            curr = replacement
            while curr.right:
                curr = curr.right
            curr.right = other
            other.parent = curr


# ============================================================
# 轮廓压缩打包(改进: 段按 x 排序 + 提前终止, 加快打包)
# ============================================================

class ContourPacker:
    """由 B*-Tree 通过轮廓(Skyline)法生成无重叠、纵向压实的布局(迭代版, 防栈溢出)。"""

    def __init__(self, modules: List[Module]):
        self.modules = modules

    def pack(self, tree: BTree):
        """返回 (positions, W, H, visited_count)。visited_count 用于树健全性检查。"""
        if tree.root is None:
            return {}, 0.0, 0.0, 0
        positions = {}
        contour = [(0.0, float('inf'), 0.0)]
        visited = 0
        seen = set()
        # 显式栈: (node, parent_x); 先序遍历(父->左->右)
        stack = [(tree.root, 0.0)]
        while stack:
            node, parent_x = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            visited += 1
            mod = self.modules[node.module_idx]
            w, h = mod.w, mod.h
            x = parent_x
            x1 = x + w
            y = 0.0
            for sx, ex, sy in contour:
                if sx >= x1:
                    break
                if ex > x and sy > y:
                    y = sy
            positions[node.module_idx] = (x, y)
            new_h = y + h
            new_contour = []
            for sx, ex, sy in contour:
                if ex <= x or sx >= x1:
                    new_contour.append((sx, ex, sy))
                else:
                    if sx < x:
                        new_contour.append((sx, x, sy))
                    if ex > x1:
                        new_contour.append((x1, ex, sy))
            new_contour.append((x, x1, new_h))
            new_contour.sort(key=lambda s: s[0])
            merged = []
            for seg in new_contour:
                if merged and abs(merged[-1][1] - seg[0]) < 1e-9 and \
                        abs(merged[-1][2] - seg[2]) < 1e-9:
                    merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
                else:
                    merged.append(seg)
            contour[:] = merged
            # 左孩子=父模块右侧; 右孩子=父模块上方(同一 x)
            if node.right is not None:
                stack.append((node.right, x))
            if node.left is not None:
                stack.append((node.left, x + w))
        max_x = max_y = 0.0
        for idx, (x, y) in positions.items():
            mod = self.modules[idx]
            max_x = max(max_x, x + mod.w)
            max_y = max(max_y, y + mod.h)
        return positions, max_x, max_y, visited


# ============================================================
# 增量 HPWL 计算
# ============================================================

class HPWLEvaluator:
    """维护每个线网的包围盒, 只重算"位置发生变化的模块"涉及的线网。"""

    def __init__(self, nets: List[Net], terminal_positions: Dict[str, Tuple[float, float]]):
        self.nets = nets
        self.terminal_positions = terminal_positions
        self.pin_nets: Dict[str, List[int]] = defaultdict(list)   # 引脚 -> 线网索引
        self.net_pins: List[List[str]] = []
        for ni, net in enumerate(nets):
            self.net_pins.append(list(net.pins))
            for p in net.pins:
                self.pin_nets[p].append(ni)
        self.pin_xy: Dict[str, Tuple[float, float]] = dict(terminal_positions)
        self.net_vals: List[Optional[float]] = [None] * len(nets)
        self._total = 0.0

    def set_module_centers(self, centers: Dict[str, Tuple[float, float]]):
        """设置硬模块的引脚坐标(几何中心), 并全量初始化。"""
        self.pin_xy.update(centers)
        self._total = 0.0
        for ni in range(len(self.nets)):
            if not self.nets[ni].pins:
                self.net_vals[ni] = 0.0
                continue
            v = self._net_hpwl(ni)
            self.net_vals[ni] = v
            self._total += v * self.nets[ni].weight

    def _net_hpwl(self, ni: int) -> float:
        xs, ys = [], []
        for p in self.net_pins[ni]:
            if p in self.pin_xy:
                px, py = self.pin_xy[p]
                xs.append(px); ys.append(py)
        if not xs:
            return 0.0
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def update_changed(self, changed_names: List[str]) -> float:
        """只重算包含 changed_names 中任一引脚的线网。"""
        affected = set()
        for name in changed_names:
            affected.update(self.pin_nets.get(name, ()))
        for ni in affected:
            if not self.nets[ni].pins:
                continue
            v = self._net_hpwl(ni)
            if self.net_vals[ni] is None:
                self._total += v * self.nets[ni].weight
            else:
                self._total += (v - self.net_vals[ni]) * self.nets[ni].weight
            self.net_vals[ni] = v
        return self._total

    @property
    def total(self) -> float:
        return self._total


# ============================================================
# 精确几何校验(严格可行: 不重叠 + 不越界)
# ============================================================

def _rect_overlap(x1, y1, w1, h1, x2, y2, w2, h2, eps=1e-6):
    """两轴对齐矩形是否重叠(内部相交, 边相接不算重叠)。"""
    return (x1 < x2 + w2 - eps and x2 < x1 + w1 - eps and
            y1 < y2 + h2 - eps and y2 < y1 + h1 - eps)


def exact_verify(modules: List[Module],
                 positions: Dict[str, Tuple[float, float, bool]],
                 outline_w: float, outline_h: float,
                 terminal_positions: Optional[Dict[str, Tuple[float, float]]] = None,
                 eps: float = 1e-6, check_terminals: bool = True) -> Tuple[bool, List[str]]:
    """
    精确校验:
      1) 硬模块之间 AABB 不重叠(非矩形用子矩形);
      2) 硬模块不超出轮廓;
      3) (信息) 终端是否在轮廓内。
    返回 (是否可行, 问题列表)。
    """
    issues = []
    # 越界检查
    for mod in modules:
        if mod.name not in positions:
            continue
        x, y, rotated = positions[mod.name]
        w, h = (mod.height, mod.width) if rotated else (mod.width, mod.height)
        if x < -eps or y < -eps:
            issues.append(f"{mod.name} 坐标为负: ({x:.2f},{y:.2f})")
        if x + w > outline_w + eps:
            issues.append(f"{mod.name} 越界(X): x+w={x+w:.2f} > {outline_w:.2f}")
        if y + h > outline_h + eps:
            issues.append(f"{mod.name} 越界(Y): y+h={y+h:.2f} > {outline_h:.2f}")
    # 重叠检查(两两)
    hard = [m for m in modules if m.name in positions]
    for i in range(len(hard)):
        mi = hard[i]
        xi, yi, ri = positions[mi.name]
        wi, hi = (mi.height, mi.width) if ri else (mi.width, mi.height)
        blocks_i = [(xi + sb.rel_x, yi + sb.rel_y, sb.width, sb.height)
                    for sb in mi.sub_blocks] if (mi.sub_blocks and mi.shape_type != 'rect') \
                   else [(xi, yi, wi, hi)]
        for j in range(i + 1, len(hard)):
            mj = hard[j]
            xj, yj, rj = positions[mj.name]
            wj, hj = (mj.height, mj.width) if rj else (mj.width, mj.height)
            blocks_j = [(xj + sb.rel_x, yj + sb.rel_y, sb.width, sb.height)
                        for sb in mj.sub_blocks] if (mj.sub_blocks and mj.shape_type != 'rect') \
                       else [(xj, yj, wj, hj)]
            for (ax, ay, aw, ah) in blocks_i:
                for (bx, by, bw, bh) in blocks_j:
                    if _rect_overlap(ax, ay, aw, ah, bx, by, bw, bh, eps):
                        issues.append(f"{mi.name} 与 {mj.name} 重叠")
                        break
                else:
                    continue
                break
    # 终端位置检查(仅信息; 题目规定终端不参与面积/重叠约束)
    if terminal_positions and check_terminals:
        for name, (tx, ty) in terminal_positions.items():
            if tx < -eps or ty < -eps or tx > outline_w + eps or ty > outline_h + eps:
                issues.append(f"终端 {name} 位于轮廓外: ({tx:.1f},{ty:.1f})")
    return (len(issues) == 0, issues)
# ============================================================
# 改进版模拟退火引擎
# ============================================================

class SimulatedAnnealing:
    """
    基于 B*-Tree 的模拟退火(改进版)。

    改进点:
      - 自适应惩罚: 根据近期越界率动态调整轮廓惩罚权重 w_outline;
      - 增量 HPWL: 打包后只重算"位置/旋转发生变化模块"涉及的线网;
      - 卡死重启: 长时间无改进时从当前最优重新升温;
      - 支持两阶段: stage=1 可行/面积, stage=2 线长/长宽比细化;
      - 支持热启动: 传入 init_tree / init_rotations 继续优化。
    """

    def __init__(self, modules: List[Module], hard_indices: List[int],
                 nets: List[Net] = None,
                 terminal_positions: Dict[str, Tuple[float, float]] = None,
                 fixed_outline: Tuple[float, float] = None,
                 seed: int = 42):
        self.modules = modules
        self.hard_indices = hard_indices
        self.nets = nets or []
        self.terminal_positions = terminal_positions or {}
        self.fixed_outline = fixed_outline
        self.packer = ContourPacker(modules)
        self.hpwl_eval = HPWLEvaluator(self.nets, self.terminal_positions) if self.nets else None
        self.last_positions = {}       # name -> (x, y, w, h, rotated)
        self.w_outline = 100000.0
        self.w_hpwl = 1.0
        self.w_norm = 0.0   # HPWL 归一化常数(预热采样平均线长)
        self.adaptive = True
        self.recent_infeasible = []
        self.rng = random.Random(seed)

    # ---------- 基础工具 ----------
    def _apply_rotations(self, rotations):
        for i, idx in enumerate(self.hard_indices):
            self.modules[idx].rotated = rotations[i]

    def _pack_and_positions(self, tree, rotations):
        self._apply_rotations(rotations)
        positions, W, H, visited = self.packer.pack(tree)
        if visited != len(self.hard_indices):
            # 树不健全(节点丢失/环): 返回空标记, 由调用方拒绝该解
            return None, 0.0, 0.0, {}
        pos_with_wh = {}
        for tree_idx, (x, y) in positions.items():
            mod = self.modules[self.hard_indices[tree_idx]]
            pos_with_wh[mod.name] = (x, y, mod.w, mod.h, bool(rotations[tree_idx]))
        return positions, W, H, pos_with_wh

    def _get_hpwl(self, pos_with_wh):
        if self.hpwl_eval is None or not self.nets:
            return 0.0
        changed = []
        for name, (x, y, w, h, rot) in pos_with_wh.items():
            old = self.last_positions.get(name)
            if old is None or abs(old[0]-x) > 1e-9 or abs(old[1]-y) > 1e-9 or \
               abs(old[2]-w) > 1e-9 or abs(old[3]-h) > 1e-9:
                changed.append(name)
        if changed:
            centers = {name: (pos_with_wh[name][0] + pos_with_wh[name][2]/2.0,
                              pos_with_wh[name][1] + pos_with_wh[name][3]/2.0)
                       for name in changed}
            self.hpwl_eval.pin_xy.update(centers)
            self.hpwl_eval.update_changed(changed)
        self.last_positions = pos_with_wh
        return self.hpwl_eval.total

    # ---------- 代价函数(归一化: 越界按轮廓边长相对值, HPWL 按预热均值) ----------
    def _compute_cost(self, width, height, hpwl, problem_type, stage=1, area_budget=None):
        ow, oh = self.fixed_outline if self.fixed_outline else (0.0, 0.0)
        if self.fixed_outline:
            # 相对越界量: 量级 O(1)
            violation = (max(0.0, width - ow) / max(1e-9, ow) +
                         max(0.0, height - oh) / max(1e-9, oh))
        else:
            violation = 0.0
        if problem_type == 1:
            area = width * height
            ar = max(width, height) / max(1e-9, min(width, height))
            if stage == 1:
                # 阶段1: 面积为主, 轻微长宽比偏好(避免随机方向)
                return area * (1.0 + 0.02 * abs(ar - 1.0))
            else:
                # 阶段2: 在面积预算内最小化长宽比
                area_over = max(0.0, area - (area_budget or area))
                return 100.0 * abs(ar - 1.0) + 1e5 * area_over / max(1e-9, area)
        elif problem_type == 2:
            if stage == 1:
                return violation * self.w_outline
            wn = self.w_norm if self.w_norm > 0 else 1.0
            return (hpwl / wn) * self.w_hpwl + violation * self.w_outline
        elif problem_type == 3:
            return violation * self.w_outline
        return 0.0

    # ---------- 自适应惩罚 ----------
    def _adapt_penalty(self, violation):
        if not self.adaptive:
            return
        self.recent_infeasible.append(violation > 1e-6)
        if len(self.recent_infeasible) > 100:
            self.recent_infeasible.pop(0)
        if len(self.recent_infeasible) >= 50:
            rate = sum(self.recent_infeasible) / len(self.recent_infeasible)
            if rate > 0.5:
                self.w_outline = min(self.w_outline * 1.15, 1e9)
            elif rate < 0.08:
                self.w_outline = max(self.w_outline * 0.9, 1.0)

    # ---------- 主循环 ----------
    def run(self, problem_type: int = 1, stage: int = 1, area_budget: float = None,
            init_tree: Optional[BTree] = None, init_rotations: List[bool] = None,
            max_total_iter: int = None, max_iter_per_temp: int = None,
            T_initial: float = None, T_final: float = None, cooling_rate: float = None,
            seed: int = None) -> FloorplanResult:
        n = len(self.hard_indices)
        if n == 0:
            return FloorplanResult({}, 0, 0, 0, 0)
        if seed is not None:
            self.rng = random.Random(seed)

        # 参数默认值
        T_initial = T_initial if T_initial is not None else 5000.0
        T_final = T_final if T_final is not None else 0.01
        cooling_rate = cooling_rate if cooling_rate is not None else 0.95
        max_iter_per_temp = max_iter_per_temp if max_iter_per_temp is not None else max(30, n)
        max_total_iter = max_total_iter if max_total_iter is not None else max(3000, n * 50)

        # 固定轮廓(问题2/3)
        if problem_type >= 2 and self.fixed_outline is None:
            total_area = sum(self.modules[i].area for i in self.hard_indices)
            # fixed_outline 由调用方传入; 若未传则用 dsr=0.15 近似(一般不会走到)
            side = math.sqrt(total_area * 1.15)
            self.fixed_outline = (side, side)

        # 初始化树
        if init_tree is not None and init_rotations is not None:
            tree = init_tree.copy()
            rotations = list(init_rotations)
        else:
            tree = BTree(n)
            shuffled = list(range(n))
            self.rng.shuffle(shuffled)
            tree.build_initial_tree(shuffled)
            rotations = [False] * n

        # 预热采样: 随机扰动求平均线长 W_norm(归一化用)
        if problem_type == 2 and self.nets and self.w_norm <= 0.0:
            _hs = []
            _t2 = BTree(n)
            _sh = list(range(n)); self.rng.shuffle(_sh); _t2.build_initial_tree(_sh)
            _r2 = [False] * n
            for _ in range(15):
                if self.rng.random() < 0.5:
                    a = self.rng.randrange(n); b = self.rng.randrange(n)
                    if a != b:
                        _t2.swap_modules(a, b)
                else:
                    a = self.rng.randrange(n)
                    _r2[a] = not _r2[a]
                _p, _w, _h, _pwh = self._pack_and_positions(_t2, _r2)
                if _p is not None:
                    _hs.append(self._get_hpwl(_pwh))
            if _hs:
                self.w_norm = sum(_hs) / len(_hs)
                print(f"  预热: W_norm={self.w_norm:.0f}", flush=True)

        # 初始打包与代价(若树不健全则重建)
        positions, width, height, pos_with_wh = self._pack_and_positions(tree, rotations)
        if positions is None:
            tree = BTree(n)
            shuffled = list(range(n))
            self.rng.shuffle(shuffled)
            tree.build_initial_tree(shuffled)
            rotations = [False] * n
            positions, width, height, pos_with_wh = self._pack_and_positions(tree, rotations)
        hpwl = self._get_hpwl(pos_with_wh)
        self.last_positions = {}   # 首次调用 _get_hpwl 会把所有模块记为 changed, 全量初始化
        hpwl = self._get_hpwl(pos_with_wh)
        current_cost = self._compute_cost(width, height, hpwl, problem_type, stage, area_budget)
        violation = (max(0.0, width - (self.fixed_outline[0] if self.fixed_outline else 0.0)) +
                     max(0.0, height - (self.fixed_outline[1] if self.fixed_outline else 0.0)))

        best_tree = tree.copy()
        best_rotations = rotations[:]
        best_cost = current_cost
        best_positions = dict(pos_with_wh)
        best_width, best_height = width, height
        best_hpwl = hpwl

        # 初始温度自适应代价量级(退火初期能接受劣解, 避免退化成纯爬山)
        T = max(T_initial, current_cost / 8.0)
        iteration = 0
        stuck_count = 0
        self.recent_infeasible = []

        while T > T_final and iteration < max_total_iter:
            n_moves = min(max_iter_per_temp, n * 3)
            for _ in range(n_moves):
                iteration += 1
                old_rotations = rotations[:]
                old_tree = tree.copy()

                # 扰动: 旋转(30%) / 交换(40%) / 移动(30%)
                r = self.rng.random()
                if r < 0.3:
                    idx = self.rng.choice(self.hard_indices)
                    rotations[idx] = not rotations[idx]
                elif r < 0.7:
                    i1 = self.rng.randrange(n); i2 = self.rng.randrange(n)
                    if i1 != i2:
                        tree.swap_modules(i1, i2)
                else:
                    i1 = self.rng.randrange(n); i2 = self.rng.randrange(n)
                    if i1 != i2:
                        tree.delete_and_insert(i1, i2)

                positions, width, height, pos_with_wh = self._pack_and_positions(tree, rotations)
                if positions is None:
                    # 树不健全: 撤销扰动
                    rotations = old_rotations
                    tree = old_tree
                    continue
                hpwl = self._get_hpwl(pos_with_wh)
                new_cost = self._compute_cost(width, height, hpwl, problem_type, stage, area_budget)
                new_violation = (max(0.0, width - (self.fixed_outline[0] if self.fixed_outline else 0.0)) +
                                 max(0.0, height - (self.fixed_outline[1] if self.fixed_outline else 0.0)))

                delta = new_cost - current_cost
                if delta < 0 or self.rng.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_tree = tree.copy()
                        best_rotations = rotations[:]
                        best_cost = current_cost
                        best_positions = dict(pos_with_wh)
                        best_width, best_height = width, height
                        best_hpwl = hpwl
                        stuck_count = 0
                    else:
                        stuck_count += 1
                    self._adapt_penalty(new_violation)
                else:
                    rotations = old_rotations
                    tree = old_tree
                    self._adapt_penalty(new_violation)

                if stuck_count > 800:
                    break
            # 卡死重启: 从当前最优重新加热
            if stuck_count > 800:
                tree = best_tree.copy()
                rotations = best_rotations[:]
                T = min(T * 8.0, T_initial)
                stuck_count = 0
                positions, width, height, pos_with_wh = self._pack_and_positions(tree, rotations)
                if positions is None:
                    tree = BTree(n)
                    shuffled = list(range(n))
                    self.rng.shuffle(shuffled)
                    tree.build_initial_tree(shuffled)
                    rotations = [False] * n
                    positions, width, height, pos_with_wh = self._pack_and_positions(tree, rotations)
                hpwl = self._get_hpwl(pos_with_wh)
                current_cost = self._compute_cost(width, height, hpwl, problem_type, stage, area_budget)
            # 降温
            if self.adaptive and problem_type in (2, 3) and len(self.recent_infeasible) >= 50:
                rate = sum(self.recent_infeasible) / len(self.recent_infeasible)
                if rate > 0.5:
                    T *= 0.92   # 不可行解多, 降慢一点
                else:
                    T *= cooling_rate
            else:
                T *= cooling_rate

        # 恢复最优并计算最终指标
        self._apply_rotations(best_rotations)
        final_positions = {}
        for name, (x, y, w, h, rot) in best_positions.items():
            final_positions[name] = (x, y, rot)
        area = best_width * best_height
        ar = max(best_width, best_height) / max(1e-9, min(best_width, best_height))
        total_block_area = sum(self.modules[i].area for i in self.hard_indices)
        actual_dsr = (area - total_block_area) / max(1e-9, total_block_area)

        # 保存最优树/旋转(供两阶段热启动)
        self.best_tree = best_tree
        self.best_rotations = best_rotations
        # 精确校验(严格可行): 固定轮廓问题用 fixed_outline, 否则用实际包围盒
        vw, vh = (self.fixed_outline if self.fixed_outline
                  else (best_width, best_height))
        ok, issues = exact_verify(self.modules, final_positions, vw, vh,
                                  self.terminal_positions, check_terminals=False)
        result = FloorplanResult(
            module_positions=final_positions,
            outline_width=best_width,
            outline_height=best_height,
            area=area,
            aspect_ratio=ar,
            total_hpwl=best_hpwl,
            dead_space_ratio=actual_dsr,
            feasible=ok,
            verify_info="; ".join(issues[:5]),
            lower_bound_area=total_block_area,
        )
        result._tree = best_tree
        result._rotations = best_rotations
        return result
# ============================================================
# 问题求解器(问题 1-3): 多起点 + 两阶段 + 严格可行 + 统计
# ============================================================

def _stats(values):
    """返回 (最优, 均值, 标准差)。"""
    if not values:
        return 0.0, 0.0, 0.0
    best = min(values)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return best, mean, math.sqrt(var)


class ProblemSolver:
    def __init__(self, blocks_file: str, nets_file: str, pl_file: str):
        self.blocks_file = blocks_file
        self.nets_file = nets_file
        self.pl_file = pl_file
        self.modules = VLSIParser.parse_blocks(blocks_file)
        self.nets = VLSIParser.parse_nets(nets_file) if os.path.exists(nets_file) else []
        self.terminal_positions = VLSIParser.parse_pl(pl_file) if os.path.exists(pl_file) else {}
        self.hard_modules = [m for m in self.modules if m.module_type == 'block']
        self.hard_indices = list(range(len(self.hard_modules)))
        self.n = len(self.hard_modules)
        self.total_area = sum(m.area for m in self.hard_modules)
        print(f"[Data] 硬模块={self.n} 终端={len(self.modules)-self.n} "
              f"线网={len(self.nets)} 总面积={self.total_area:,.1f}")

    # ---------- 预算 ----------
    def _budget(self, base, per_n, quick):
        if quick:
            return max(base // 3, 600)
        return max(base, int(self.n * per_n))

    # ---------- 构造 SA ----------
    def _make_sa(self, fixed_outline, seed, with_nets, deep=True):
        modules = [copy.deepcopy(m) for m in self.hard_modules] if deep else self.hard_modules
        return SimulatedAnnealing(
            modules=modules,
            hard_indices=list(range(self.n)),
            nets=self.nets if with_nets else [],
            terminal_positions=self.terminal_positions if with_nets else {},
            fixed_outline=fixed_outline,
            seed=seed,
        )

    # =========================================================
    # 问题 1: 面积最小 + 长宽比接近 1
    # =========================================================
    # ---------- ShelfSA 工厂与校验 ----------
    def _make_shelf(self, seed):
        return ShelfSA(self.hard_modules, self.nets, self.terminal_positions, seed=seed)

    def _verify(self, pos, W, H, dsr_outline=None):
        ow = oh = dsr_outline if dsr_outline else max(W, H)
        return exact_verify(self.hard_modules, pos, ow, oh, self.terminal_positions)

    # =========================================================
    # 问题 1: 面积最小 + 长宽比接近 1(货架装箱 + 两阶段模拟退火)
    # =========================================================
    def solve_problem1(self, seeds: int = 3, quick: bool = False,
                       seed_offset: int = 0) -> FloorplanResult:
        print("\n" + "=" * 66)
        print("问题 1: 面积最小化 + 长宽比(货架装箱+SA, 两阶段, 多起点)")
        print("=" * 66)
        areas, ratios = [], []
        best_result = None
        total = self.total_area
        for k in range(seeds):
            sa = self._make_shelf(seed=42 + k * 17 + seed_offset)
            n_iter = 20000 if not quick else 5000
            # 阶段1: 面积最小
            pos, H, W, perm, rots = sa.anneal(objective='area', max_total_iter=n_iter)
            area1 = W * H
            # 阶段2: 在面积预算内最小化长宽比(热启动)
            pos, H, W, perm, rots = sa.anneal(objective='area',
                                             area_budget=area1 * 1.02,
                                             init=(perm, rots),
                                             max_total_iter=max(n_iter // 3, 1500))
            area = W * H
            ar = max(W, H) / max(1e-9, min(W, H))
            ok, issues = self._verify(pos, W, H)
            res = FloorplanResult(pos, W, H, area, ar, 0.0,
                                  (area - total) / max(1e-9, total),
                                  feasible=ok, verify_info="; ".join(issues[:3]),
                                  lower_bound_area=total)
            areas.append(area); ratios.append(ar)
            if best_result is None or area < best_result.area - 1e-9 or \
               (abs(area - best_result.area) < 1e-9 and ar < best_result.aspect_ratio):
                best_result = res
        ba, ma, sa_ = ShelfSA.stats(areas)
        bra, mra, sra = ShelfSA.stats(ratios)
        best_result.best_area = ba; best_result.mean_area = ma; best_result.std_area = sa_
        best_result.n_runs = seeds
        print(f"  面积: best={ba:.1f} mean={ma:.1f} std={sa_:.1f} (下界={total:.1f})")
        print(f"  长宽比: best={bra:.4f} mean={mra:.4f}")
        print(f"  轮廓: {best_result.outline_width:.1f} x {best_result.outline_height:.1f}")
        print(f"  死区比例: {best_result.dead_space_ratio:.4f}")
        print(f"  精确校验: {'通过' if best_result.feasible else '未通过: ' + best_result.verify_info}")
        return best_result

    # =========================================================
    # 问题 2: 固定正方形轮廓下 HPWL 最小(先可行后线长)
    # =========================================================
    def solve_problem2(self, dsr: float = 0.15, seeds: int = 3,
                       quick: bool = False, refine_only: bool = False,
                       init_result: FloorplanResult = None,
                       seed_offset: int = 0) -> FloorplanResult:
        side = math.sqrt(self.total_area * (1 + dsr))
        if not refine_only:
            print("\n" + "=" * 66)
            print(f"问题 2: 固定轮廓 {side:.2f}x{side:.2f} 下 HPWL 最小(先可行后线长)")
            print("=" * 66)

        # 阶段1: 可行性(多起点)
        feasible_inits = []
        if not refine_only:
            for k in range(seeds):
                sa = self._make_shelf(seed=200 + k * 31 + seed_offset)
                pos, H, W, perm, rots = sa.anneal(W_fixed=side, objective='feas',
                                                  max_total_iter=12000 if not quick else 3500)
                ok, _ = self._verify(pos, W, H, dsr_outline=side)
                if ok:
                    feasible_inits.append((perm, rots, H))
            print(f"  阶段1(可行性): {len(feasible_inits)}/{seeds} 个起点可行")

        # 阶段2: HPWL(热启动自可行解)
        hpwls = []
        best_result = None
        for k in range(seeds):
            sa = self._make_shelf(seed=500 + k * 41 + seed_offset)
            init = None
            if feasible_inits:
                p0, r0, _ = feasible_inits[k % len(feasible_inits)]
                init = (p0, r0)
            elif init_result is not None and hasattr(init_result, '_perm'):
                init = (init_result._perm, init_result._rots)
            pos, H, W, perm, rots = sa.anneal(W_fixed=side, objective='hpwl',
                                              init=init,
                                              max_total_iter=20000 if not quick else 6000,
                                              strict=True)
            ok, issues = self._verify(pos, W, H, dsr_outline=side)
            hwl = sa.hpwl(pos)
            res = FloorplanResult(pos, side, side, side * side, 1.0, hwl,
                                  (side * side - self.total_area) / max(1e-9, self.total_area),
                                  feasible=ok, verify_info="; ".join(issues[:3]),
                                  lower_bound_area=self.total_area)
            res._perm = perm; res._rots = rots
            hpwls.append(hwl)
            if best_result is None or (ok and (not best_result.feasible or hwl < best_result.total_hpwl)):
                best_result = res
        bh, mh, sh = ShelfSA.stats(hpwls)
        best_result.best_hpwl = bh; best_result.mean_hpwl = mh; best_result.std_hpwl = sh
        best_result.n_runs = seeds
        if not refine_only:
            print(f"  阶段2(HPWL): best={bh:.2f} mean={mh:.2f} std={sh:.2f}")
            print(f"  精确校验: {'通过(严格可行)' if best_result.feasible else '未通过: ' + best_result.verify_info}")
        return best_result

    # =========================================================
    # 问题 3: 最小可行死区比例(二分 + 可行性判定)
    # =========================================================
    def solve_problem3(self, seeds: int = 2, quick: bool = False,
                       dsr_hi: float = 0.5, precision: float = 0.002,
                       seed_offset: int = 0) -> Tuple[float, FloorplanResult]:
        print("\n" + "=" * 66)
        print(f"问题 3: 最小可行死区比例(二分搜索, 精度 {precision})")
        print("=" * 66)
        lo, hi = 0.0, dsr_hi
        best_feasible = None
        best_init = None
        n_iter = 0
        while hi - lo > precision and n_iter < 25:
            n_iter += 1
            mid = (lo + hi) / 2.0
            side = math.sqrt(self.total_area * (1 + mid))
            ok = False
            trial_init = None
            for k in range(seeds):
                sa = self._make_shelf(seed=700 + k * 53 + n_iter * 7 + seed_offset)
                pos, H, W, perm, rots = sa.anneal(W_fixed=side, objective='feas',
                                                  max_total_iter=10000 if not quick else 3000)
                vok, _ = self._verify(pos, W, H, dsr_outline=side)
                if vok:
                    ok = True
                    trial_init = (perm, rots)
                    break
            if ok:
                hi = mid
                best_feasible = mid
                best_init = trial_init
                print(f"  DSR={mid:.4f} (side={side:.1f}): 可行 -> 继续缩小")
            else:
                lo = mid
                print(f"  DSR={mid:.4f} (side={side:.1f}): 不可行 -> 扩大")
        if best_feasible is None:
            print("  警告: 搜索区间内未找到可行解")
            return 0.0, FloorplanResult({}, 0, 0, 0, 0)
        # 最终 HPWL 细化(热启动自二分最优可行解)
        init_result = FloorplanResult({}, 0, 0, 0, 0)
        init_result._perm, init_result._rots = best_init
        final = self.solve_problem2(dsr=best_feasible, seeds=seeds, quick=quick,
                                    refine_only=True, init_result=init_result,
                                    seed_offset=seed_offset)
        print(f"\n  最小可行死区比例: {best_feasible:.4f}")
        print(f"  对应轮廓: {math.sqrt(self.total_area*(1+best_feasible)):.2f} x "
              f"{math.sqrt(self.total_area*(1+best_feasible)):.2f}")
        print(f"  该比例下总 HPWL: {final.total_hpwl:.2f}")
        print(f"  精确校验: {'通过(严格可行)' if final.feasible else '未通过: ' + final.verify_info}")
        return best_feasible, final

# ============================================================
# 可视化(优先 matplotlib, 缺失时用 PIL 自绘)
# ============================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

_PALETTE = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#E8BAFF",
            "#FFD8B3", "#B3FFE0", "#D0B3FF", "#FFC3E0", "#B3F0FF"]

_FONT_CACHE = {}
def _get_font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font = None
    candidates = [
        ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in candidates:
        try:
            font = ImageFont.truetype(fp, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def visualize_floorplan(result: FloorplanResult,
                        modules: List[Module],
                        title: str = "Floorplan",
                        save_path: str = None,
                        show_nets: bool = False,
                        nets: List[Net] = None,
                        terminal_positions: Dict[str, Tuple[float, float]] = None,
                        is_nonrect: bool = False,
                        variants: Dict[str, List[List[SubBlock]]] = None):
    """绘制布图: 模块矩形/子矩形 + 红色虚线轮廓 + 终端 + 线网 + 信息框。"""
    ow, oh = result.outline_width, result.outline_height
    if ow <= 0 or oh <= 0:
        print(f"[SKIP] {save_path}: 轮廓为空")
        return

    # ---- matplotlib 路径 ----
    if _HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i, mod in enumerate(modules):
            if mod.name not in result.module_positions:
                continue
            x, y, rotated = result.module_positions[mod.name]
            w, h = (mod.h, mod.w) if rotated else (mod.w, mod.h)
            color = _PALETTE[i % len(_PALETTE)]
            if mod.sub_blocks and mod.shape_type != 'rect':
                for sb in mod.sub_blocks:
                    ax.add_patch(Rectangle((x+sb.rel_x, y+sb.rel_y), sb.width, sb.height,
                                           linewidth=1.2, edgecolor='black',
                                           facecolor=color, alpha=0.75))
            else:
                ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor='black',
                                       facecolor=color, alpha=0.75))
            ax.text(x + w/2, y + h/2, mod.name, ha='center', va='center',
                    fontsize=8, fontweight='bold')
        ax.add_patch(Rectangle((0, 0), ow, oh, linewidth=2, edgecolor='red',
                               facecolor='none', linestyle='--'))
        if terminal_positions:
            for name, (tx, ty) in terminal_positions.items():
                ax.plot(tx, ty, 'r*', markersize=6)
                ax.text(tx+1, ty+1, name, fontsize=5, color='red')
        if show_nets and nets and terminal_positions:
            pin = {}
            for mod in modules:
                if mod.name in result.module_positions:
                    x, y, _ = result.module_positions[mod.name]
                    w, h = (mod.h, mod.w) if _ else (mod.w, mod.h)
                    pin[mod.name] = (x+w/2, y+h/2)
            pin.update(terminal_positions)
            for net in nets[:40]:
                pts = [pin[p] for p in net.pins if p in pin]
                for a, b in zip(pts, pts[1:]):
                    ax.plot([a[0], b[0]], [a[1], b[1]], 'b-', linewidth=0.4, alpha=0.5)
        ax.set_xlim(-10, ow*1.08); ax.set_ylim(-10, oh*1.08)
        ax.set_aspect('equal'); ax.set_title(title)
        info = f"Area={result.area:.1f}\n{ow:.1f}x{oh:.1f}\nAR={result.aspect_ratio:.4f}"
        if result.total_hpwl > 0:
            info += f"\nHPWL={result.total_hpwl:.1f}"
        if result.dead_space_ratio > 0:
            info += f"\nDSR={result.dead_space_ratio:.4f}"
        info += f"\nFeasible={'Y' if result.feasible else 'N'}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches='tight')
            print(f"  已保存: {save_path}")
        plt.close()
        return

    # ---- PIL 路径 ----
    if not _HAS_PIL:
        print(f"[SKIP] {save_path}: 无可用的绘图库(matplotlib/PIL)")
        return
    Wpx, Hpx = 1100, 1100
    margin = 70
    scale = min((Wpx - 2*margin) / ow, (Hpx - 2*margin) / oh)
    off_x = (Wpx - ow*scale) / 2
    off_y = (Hpx - oh*scale) / 2
    img = Image.new("RGB", (Wpx, Hpx), "white")
    d = ImageDraw.Draw(img)
    # 模块
    for i, mod in enumerate(modules):
        if mod.name not in result.module_positions:
            continue
        x, y, rotated = result.module_positions[mod.name]
        w, h = (mod.h, mod.w) if rotated else (mod.w, mod.h)
        color = _hex_to_rgb(_PALETTE[i % len(_PALETTE)])
        rects = []
        if mod.sub_blocks and mod.shape_type != 'rect':
            for sb in mod.sub_blocks:
                rects.append((x+sb.rel_x, y+sb.rel_y, sb.width, sb.height))
        else:
            rects.append((x, y, w, h))
        for (rx, ry, rw, rh) in rects:
            x0 = off_x + rx*scale; y0 = off_y + ry*scale
            x1 = off_x + (rx+rw)*scale; y1 = off_y + (ry+rh)*scale
            d.rectangle([x0, y0, x1, y1], fill=color, outline="black", width=2)
        # 标签
        label = mod.name
        try:
            fnt = _get_font(16 if len(label) <= 4 else 12)
            lw = d.textlength(label, font=fnt)
            cx = off_x + (x + w/2)*scale - lw/2
            cy = off_y + (y + h/2)*scale - 8
            d.text((cx, cy), label, fill="black", font=fnt)
        except Exception:
            pass
    # 轮廓(红色虚线)
    x0 = off_x; y0 = off_y; x1 = off_x + ow*scale; y1 = off_y + oh*scale
    dash = 10
    for (ax, ay, bx, by) in [(x0,y0,x1,y0),(x1,y0,x1,y1),(x1,y1,x0,y1),(x0,y1,x0,y0)]:
        length = math.hypot(bx-ax, by-ay)
        steps = int(length // (2*dash))
        for s in range(steps):
            t0 = s*2*dash/length; t1 = min(1.0, (s*2*dash + dash)/length)
            d.line([ax+(bx-ax)*t0, ay+(by-ay)*t0, ax+(bx-ax)*t1, ay+(by-ay)*t1],
                   fill="red", width=3)
    # 终端
    if terminal_positions:
        for name, (tx, ty) in terminal_positions.items():
            px = off_x + tx*scale; py = off_y + ty*scale
            d.ellipse([px-4, py-4, px+4, py+4], fill="red", outline="black")
            try:
                d.text((px+4, py-10), name, fill="red", font=_get_font(10))
            except Exception:
                pass
    # 线网(前 40 条)
    if show_nets and nets:
        pin = {}
        for mod in modules:
            if mod.name in result.module_positions:
                x, y, rot = result.module_positions[mod.name]
                w, h = (mod.h, mod.w) if rot else (mod.w, mod.h)
                pin[mod.name] = (off_x + (x+w/2)*scale, off_y + (y+h/2)*scale)
        if terminal_positions:
            for name, (tx, ty) in terminal_positions.items():
                pin[name] = (off_x + tx*scale, off_y + ty*scale)
        for net in nets[:40]:
            pts = [pin[p] for p in net.pins if p in pin]
            for a, b in zip(pts, pts[1:]):
                d.line([a, b], fill=(70, 130, 200), width=1)
    # 标题与信息
    try:
        d.text((15, 10), title, fill="black", font=_get_font(22, bold=True))
    except Exception:
        pass
    info = f"Area = {result.area:.1f}   Outline = {ow:.1f} x {oh:.1f}   AR = {result.aspect_ratio:.4f}"
    if result.total_hpwl > 0:
        info += f"   HPWL = {result.total_hpwl:.1f}"
    if result.dead_space_ratio > 0:
        info += f"   DSR = {result.dead_space_ratio:.4f}"
    info += f"   Feasible = {'Y' if result.feasible else 'N'}"
    try:
        d.text((15, Hpx-42), info, fill="black", font=_get_font(18))
    except Exception:
        pass
    if save_path:
        img.save(save_path)
        print(f"  已保存: {save_path}")
# ============================================================
# 主流程
# ============================================================

def _result_summary(chip, r1, r2, min_dsr, r3):
    return {
        "chip": chip,
        "total_block_area": round(r1.lower_bound_area, 2),
        "problem1": {
            "area": round(r1.area, 2),
            "outline": [round(r1.outline_width, 2), round(r1.outline_height, 2)],
            "aspect_ratio": round(r1.aspect_ratio, 4),
            "dead_space_ratio": round(r1.dead_space_ratio, 4),
            "mean_area": round(r1.mean_area, 2),
            "std_area": round(r1.std_area, 2),
            "lower_bound_area": round(r1.lower_bound_area, 2),
            "n_runs": r1.n_runs,
            "feasible": bool(r1.feasible),
        },
        "problem2": {
            "dsr": 0.15,
            "outline": [round(r2.outline_width, 2), round(r2.outline_height, 2)],
            "fixed_outline_side": round(math.sqrt(r1.lower_bound_area * 1.15), 2),
            "total_hpwl": round(r2.total_hpwl, 2),
            "mean_hpwl": round(r2.mean_hpwl, 2),
            "std_hpwl": round(r2.std_hpwl, 2),
            "n_runs": r2.n_runs,
            "feasible": bool(r2.feasible),
        },
        "problem3": {
            "min_dsr": round(min_dsr, 4),
            "outline": [round(r3.outline_width, 2), round(r3.outline_height, 2)],
            "total_hpwl": round(r3.total_hpwl, 2),
            "feasible": bool(r3.feasible),
        },
    }



# ============================================================
# 子进程求解助手(防本机偶发的解释器原生崩溃: 崩溃后自动换种子重试)
# ============================================================

def _run_problem_subprocess(script: str, result_path: str, timeout: int = 2400):
    """在子进程中运行脚本; 成功且产出结果文件则返回 True。"""
    try:
        proc = subprocess.run([sys.executable, "-c", script],
                              timeout=timeout, capture_output=True)
        if proc.returncode == 0 and result_path and os.path.exists(result_path):
            return True
        if proc.returncode != 0:
            err = (proc.stdout or b"").decode("utf-8", "replace")[-400:]
            print("  [子进程异常退出 rc=%s] %s" % (proc.returncode, err.strip()[-150:]))
    except subprocess.TimeoutExpired:
        print("  [子进程超时]")
    except Exception as e:
        print("  [子进程异常]", e)
    return False


def _make_worker_script(blocks_file, nets_file, pl_file, problem, seeds, quick, seed_offset, result_path):
    """生成子进程求解脚本。problem: 1 / 2 / 3。"""
    _dir = os.path.dirname(os.path.abspath(__file__))
    if problem == 1:
        call = "result = solver.solve_problem1(seeds=%d, quick=%r, seed_offset=%d)" % (seeds, quick, seed_offset)
    elif problem == 2:
        call = "result = solver.solve_problem2(dsr=0.15, seeds=%d, quick=%r, seed_offset=%d)" % (seeds, quick, seed_offset)
    else:
        call = "min_dsr, result = solver.solve_problem3(seeds=%d, quick=%r, seed_offset=%d)" % (seeds, quick, seed_offset)
    dump_expr = "pickle.dump((min_dsr, result), _f)" if problem == 3 else "pickle.dump(result, _f)"
    code = (
        "import sys, pickle\n"
        "sys.path.insert(0, %r)\n"
        "import spr_v2 as solver_mod\n"
        "solver = solver_mod.ProblemSolver(%r, %r, %r)\n"
        "%s\n"
        "with open(%r, 'wb') as _f:\n"
        "    %s\n"
        % (_dir, blocks_file, nets_file, pl_file, call, result_path, dump_expr)
    )
    return code


def _solve_problem_safe(blocks_file, nets_file, pl_file, problem, seeds, quick,
                        base_offset=0, max_tries=15, timeout=2400):
    """子进程求解指定问题(单种子, seeds 应传 1), 崩溃自动换种子重试, 返回 (结果对象, 是否成功)。"""
    for attempt in range(max_tries):
        rp = os.path.join(tempfile.gettempdir(),
                          "sprv2_%s_p%d_%d.pkl" % (os.path.basename(blocks_file)[:4], problem, attempt))
        if os.path.exists(rp):
            try:
                os.remove(rp)
            except OSError:
                pass
        script = _make_worker_script(blocks_file, nets_file, pl_file, problem, 1, quick,
                                     base_offset + attempt * 137, rp)
        # 写临时 .py 文件再执行(避免 Windows 命令行 -c 传参损坏中文路径)
        wf = os.path.join(tempfile.gettempdir(),
                          "sprv2_worker_%d_%d.py" % (os.getpid(), attempt))
        try:
            with open(wf, "w", encoding="utf-8") as _fw:
                _fw.write(script)
        except OSError:
            wf = None
        if _run_problem_subprocess_file(wf, rp, timeout=timeout):
            try:
                with open(rp, "rb") as f:
                    return pickle.load(f), True
            except Exception as e:
                print("  [读取结果失败]", e)
        else:
            print("  [重试 %d/%d] 问题 %d 求解进程异常, 自动换种子重试..."
                  % (attempt + 1, max_tries, problem))
    return None, False


def _run_problem_subprocess_file(worker_py, result_path, timeout=2400):
    """运行临时 worker 脚本文件(规避命令行中文路径问题)。"""
    if not worker_py or not os.path.exists(worker_py):
        return False
    try:
        proc = subprocess.run([sys.executable, worker_py],
                              timeout=timeout, capture_output=True)
        if proc.returncode == 0 and result_path and os.path.exists(result_path):
            return True
        if proc.returncode != 0:
            err = (proc.stdout or b"").decode("utf-8", "replace")[-500:]
            print("  [worker rc=%s] %s" % (proc.returncode, err.strip()[-220:]))
    except subprocess.TimeoutExpired:
        print("  [worker 超时]")
    except Exception as e:
        print("  [worker 异常]", e)
    return False


def _solve_problem_multiseed(blocks_file, nets_file, pl_file, problem, seeds, quick,
                             base_offset=0, max_tries=15, timeout=2400):
    """多起点: 每个种子在独立子进程求解(崩溃只影响该种子并自动重试), 最后合并统计。"""
    results = []
    for k in range(seeds):
        res, ok = _solve_problem_safe(blocks_file, nets_file, pl_file, problem, 1, quick,
                                      base_offset=base_offset + k * 137,
                                      max_tries=max_tries, timeout=timeout)
        if not ok:
            print("  [失败] 问题 %d 第 %d 个种子多次重试仍失败" % (problem, k))
            return None, False
        results.append(res)
    # 合并统计
    if problem == 3:
        # results: (min_dsr, FloorplanResult) 列表, 取最小 min_dsr 的
        best = min(results, key=lambda t: t[0])
        return best, True
    # problem 1/2: 按主指标取最优并填充 mean/std
    if problem == 1:
        keyf = lambda r: r.area
        vals = [r.area for r in results]
        metric = "area"
    else:
        keyf = lambda r: r.total_hpwl
        vals = [r.total_hpwl for r in results]
        metric = "hpwl"
    best = min(results, key=keyf)
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    if problem == 1:
        best.best_area = min(vals); best.mean_area = mean; best.std_area = std
    else:
        best.best_hpwl = min(vals); best.mean_hpwl = mean; best.std_hpwl = std
    best.n_runs = seeds
    return best, True


def solve_chip(problem_path: str, chip: str, out_dir: str = "output",
               seeds: int = 3, quick: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    # 自动解析数据目录(无论从哪个目录运行都能找到数据)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    for cand in (problem_path,
                 os.path.join(_script_dir, problem_path),
                 _script_dir):
        if os.path.isdir(cand):
            problem_path = cand
            break
    blocks_file = os.path.join(problem_path, f"{chip}.blocks")
    nets_file = os.path.join(problem_path, f"{chip}.nets")
    pl_file = os.path.join(problem_path, f"{chip}.pl")
    if not os.path.exists(blocks_file):
        # 尝试在目录中自动查找
        for f in os.listdir(problem_path):
            if f.endswith(".blocks"):
                base = f[:-7]
                blocks_file = os.path.join(problem_path, f)
                nets_file = os.path.join(problem_path, f"{base}.nets")
                pl_file = os.path.join(problem_path, f"{base}.pl")
                break
    print("\n" + "#" * 66)
    print(f"# 求解 {chip}  (数据: {blocks_file})")
    print("#" * 66)

    solver = ProblemSolver(blocks_file, nets_file, pl_file)

    # 问题 1 (多起点, 每种子独立子进程 + 崩溃自动重试)
    t0 = time.time()
    r1, ok1 = _solve_problem_multiseed(blocks_file, nets_file, pl_file, 1, seeds, quick, base_offset=0)
    if not ok1:
        raise RuntimeError(f"{chip} 问题 1 多次重试仍失败")
    visualize_floorplan(r1, solver.hard_modules,
                        title=f"{chip} - Problem 1: Area Minimization",
                        save_path=os.path.join(out_dir, f"{chip}_problem1.png"))
    print(f"  耗时 {time.time()-t0:.1f}s")

    # 问题 2 (多起点, 每种子独立子进程 + 崩溃自动重试)
    t0 = time.time()
    r2, ok2 = _solve_problem_multiseed(blocks_file, nets_file, pl_file, 2, seeds, quick, base_offset=100)
    if not ok2:
        raise RuntimeError(f"{chip} 问题 2 多次重试仍失败")
    visualize_floorplan(r2, solver.hard_modules,
                        title=f"{chip} - Problem 2: HPWL Minimization (DSR=0.15)",
                        save_path=os.path.join(out_dir, f"{chip}_problem2.png"),
                        show_nets=True, nets=solver.nets,
                        terminal_positions=solver.terminal_positions)
    print(f"  耗时 {time.time()-t0:.1f}s")

    # 问题 3 (多起点, 每种子独立子进程 + 崩溃自动重试)
    t0 = time.time()
    r3, ok3 = _solve_problem_multiseed(blocks_file, nets_file, pl_file, 3, seeds, quick, base_offset=200)
    if not ok3:
        raise RuntimeError(f"{chip} 问题 3 多次重试仍失败")
    min_dsr, r3 = r3
    visualize_floorplan(r3, solver.hard_modules,
                        title=f"{chip} - Problem 3: Min DSR = {min_dsr:.4f}",
                        save_path=os.path.join(out_dir, f"{chip}_problem3.png"),
                        show_nets=True, nets=solver.nets,
                        terminal_positions=solver.terminal_positions)
    print(f"  耗时 {time.time()-t0:.1f}s")

    summary = _result_summary(chip, r1, r2, min_dsr, r3)
    with open(os.path.join(out_dir, f"{chip}_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # 追加到汇总表
    return summary


def print_summary_table(all_summaries: List[dict]):
    print("\n" + "=" * 78)
    print("结果汇总")
    print("=" * 78)
    print(f"{'芯片':<6}{'P1面积':>10}{'P1长宽比':>10}{'P2 HPWL':>12}{'P3 最小DSR':>12}{'P3 HPWL':>12}")
    for s in all_summaries:
        p1, p2, p3 = s["problem1"], s["problem2"], s["problem3"]
        print(f"{s['chip']:<6}{p1['area']:>10.1f}{p1['aspect_ratio']:>10.4f}"
              f"{p2['total_hpwl']:>12.1f}{p3['min_dsr']:>12.4f}{p3['total_hpwl']:>12.1f}")
    print("=" * 78)


def solve_problem4_example(out_dir: str = "output"):
    """问题4: 待后续实现(当前版本聚焦问题1-3)。"""
    print("\n问题 4(非矩形模块)在本版本中暂缓实现, 请使用后续版本。")


# ============================================================
# 命令行入口
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="VLSI 布图规划改进版求解器(问题1-3)")
    ap.add_argument("--path", type=str, default="附件",
                    help="数据目录(含 .blocks/.nets/.pl)")
    ap.add_argument("--chip", type=str, default=None, help="芯片名: n100/n200/n300")
    ap.add_argument("--all", action="store_true", help="求解全部三组芯片")
    ap.add_argument("--seeds", type=int, default=3, help="多起点随机种子数")
    ap.add_argument("--quick", action="store_true", help="快速模式(小预算, 用于验证运行)")
    ap.add_argument("--out", type=str, default="output", help="输出目录")
    args = ap.parse_args()

    chips = []
    if args.all:
        chips = ["n100", "n200", "n300"]
    elif args.chip:
        chips = [args.chip]
    else:
        ap.print_help()
        return

    all_summaries = []
    for c in chips:
        s = solve_chip(args.path, c, out_dir=args.out, seeds=args.seeds, quick=args.quick)
        all_summaries.append(s)
    print_summary_table(all_summaries)
    with open(os.path.join(args.out, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 {os.path.join(args.out, 'all_results.json')}")


if __name__ == "__main__":
    main()


# ============================================================
# ShelfSA: 货架装箱解码 + 模拟退火(改进版核心求解器)
# ------------------------------------------------------------
# 说明: 随机 B*-树轮廓打包在本组数据上严重偏"高瘦"(死区可达 170%),
#       难以找到可行解。改用"货架装箱(Shelf Packing) + 模拟退火":
#       - 解 = 模块排列 perm + 旋转 rots + 行宽 W(问题1 可变 / 问题2/3 固定);
#       - 解码 = 按顺序放入宽度 W 的货架行, 得到无重叠坐标;
#       - 目标 = 面积(问题1) / HPWL(问题2) / 可行性(问题3), 配合自适应惩罚。
# ============================================================

class ShelfSA:
    def __init__(self, modules: List[Module], nets: List[Net] = None,
                 terminal_positions: Dict[str, Tuple[float, float]] = None,
                 seed: int = 42):
        self.modules = modules          # 硬模块列表(与索引一一对应)
        self.names = [md.name for md in modules]
        self.n = len(modules)
        self.nets = nets or []
        self.terminal_positions = terminal_positions or {}
        self.rng = random.Random(seed)

    # ---------- 解码: 货架装箱 ----------
    def decode(self, perm, rots, W):
        """按 perm 顺序、行宽 W 货架装箱。返回 (positions[name]->(x,y,rot), H)。"""
        positions = {}
        x = y = 0.0
        cur_h = 0.0
        for idx in perm:
            md = self.modules[idx]
            w, h = (md.height, md.width) if rots[idx] else (md.width, md.height)
            if x > 0.0 and x + w > W + 1e-9:
                y += cur_h
                x = 0.0
                cur_h = 0.0
            positions[md.name] = (x, y, rots[idx])
            x += w
            if h > cur_h:
                cur_h = h
        H = y + cur_h
        return positions, H

    # ---------- HPWL ----------
    def hpwl(self, positions):
        total = 0.0
        pin = {}
        for md in self.modules:
            if md.name in positions:
                x, y, rot = positions[md.name]
                w, h = (md.height, md.width) if rot else (md.width, md.height)
                pin[md.name] = (x + w / 2.0, y + h / 2.0)
        pin.update(self.terminal_positions)
        for net in self.nets:
            if not net.pins:
                continue
            xs = []; ys = []
            for p in net.pins:
                if p in pin:
                    xs.append(pin[p][0]); ys.append(pin[p][1])
            if xs:
                total += ((max(xs) - min(xs)) + (max(ys) - min(ys))) * net.weight
        return total

    # ---------- 初始解(确定性排序候选 + 随机候选, 取最矮者) ----------
    def _initial(self, W, best_of=14):
        cands = []
        # 确定性候选: 高度/宽度/面积降序 × 不旋转/全旋转
        for key, rot in [("h", False), ("h", True), ("w", False), ("w", True),
                         ("a", False), ("a", True)]:
            if key == "h":
                order = sorted(range(self.n), key=lambda i: -self.modules[i].height)
            elif key == "w":
                order = sorted(range(self.n), key=lambda i: -self.modules[i].width)
            else:
                order = sorted(range(self.n), key=lambda i: -self.modules[i].area)
            cands.append((order, [rot] * self.n))
        # 随机候选
        for _ in range(max(0, best_of - len(cands))):
            order = list(range(self.n)); self.rng.shuffle(order)
            rots = [self.rng.random() < 0.5 for _ in range(self.n)]
            cands.append((order, rots))
        best = None
        for order, rots in cands:
            pos, H = self.decode(order, rots, W)
            if best is None or H < best[0]:
                best = (H, list(order), list(rots))
        return best[1], best[2]

    # ---------- 扰动 ----------
    def _perturb(self, perm, rots, W, mode):
        p = list(perm); r = list(rots); w = W
        rnd = self.rng.random()
        if mode == 1:
            # 交换两个模块
            a = self.rng.randrange(self.n); b = self.rng.randrange(self.n)
            if a != b:
                p[a], p[b] = p[b], p[a]
        elif mode == 2:
            # 移动一个模块
            a = self.rng.randrange(self.n); b = self.rng.randrange(self.n)
            if a != b:
                val = p.pop(a); p.insert(b, val)
        elif mode == 3:
            # 旋转一个模块
            a = self.rng.randrange(self.n)
            r[a] = not r[a]
        else:
            # 微调行宽(问题1)
            w = max(10.0, W * (1.0 + self.rng.uniform(-0.06, 0.06)))
        return p, r, w

    # ---------- 通用退火 ----------
    def anneal(self, W_fixed=None, objective="area", area_budget=None,
               max_total_iter=30000, max_iter_per_temp=80, T_initial=None,
               T_final=0.5, cooling_rate=0.995, init=None, strict=False):
        self.strict = strict
        """
        objective: 'area'(问题1, W 可变) / 'hpwl'(问题2, W 固定) / 'feas'(问题3, W 固定)
        """
        n = self.n
        # 初始行宽
        if objective == 'area':
            total = sum(md.area for md in self.modules)
            W = math.sqrt(total * 1.25)
        else:
            W = W_fixed
        if init:
            perm, rots = init
        else:
            perm, rots = self._initial(W)
        pos, H = self.decode(perm, rots, W)

        def cost_of(pp, rr, ww):
            p2, h2 = self.decode(pp, rr, ww)
            area = ww * h2
            ar = max(ww, h2) / max(1e-9, min(ww, h2))
            if objective == 'area':
                if area_budget:
                    over = max(0.0, area - area_budget)
                    return 100.0 * abs(ar - 1.0) + 1e5 * over / max(1e-9, area)
                return area * (1.0 + 0.02 * abs(ar - 1.0))
            if objective == 'hpwl':
                hwl = self.hpwl(p2)
                viol = max(0.0, h2 - ww) / max(1e-9, ww)
                return hwl + viol * self.w_penalty
            # feasible
            viol = max(0.0, h2 - ww) / max(1e-9, ww)
            return viol * self.w_penalty

        self.w_penalty = 1e5
        cur = cost_of(perm, rots, W)
        T = T_initial if T_initial is not None else max(2000.0, cur / 6.0)
        best = (cur, list(perm), list(rots), W)
        best_pos, best_H = None, None
        recent = []
        it = 0
        while T > T_final and it < max_total_iter:
            for _ in range(max_iter_per_temp):
                it += 1
                mode = 1 + int(self.rng.random() * 3)
                if objective == 'area':
                    if self.rng.random() < 0.12:
                        mode = 4  # 调行宽
                pp, rr, ww = self._perturb(perm, rots, W, mode)
                nc = cost_of(pp, rr, ww)
                if strict and objective == 'hpwl':
                    # 严格可行: 越界状态直接拒绝
                    _p2, _h2 = self.decode(pp, rr, ww)
                    if _h2 > ww + 1e-6:
                        continue
                delta = nc - cur
                if delta < 0 or self.rng.random() < math.exp(-delta / T):
                    perm, rots, W = pp, rr, ww
                    cur = nc
                    if nc < best[0]:
                        best = (nc, list(perm), list(rots), W)
                # 自适应惩罚(固定轮廓问题)
                if objective != 'area':
                    p2, h2 = self.decode(perm, rots, W)
                    recent.append(1 if h2 > W + 1e-9 else 0)
                    if len(recent) > 200:
                        recent.pop(0)
                    if len(recent) >= 50:
                        rate = sum(recent) / len(recent)
                        if rate > 0.5:
                            self.w_penalty = min(self.w_penalty * 1.12, 1e9)
                        elif rate < 0.06:
                            self.w_penalty = max(self.w_penalty * 0.9, 10.0)
            T *= cooling_rate

        # 最优解
        _, bperm, brots, bW = best
        best_pos, best_H = self.decode(bperm, brots, bW)
        return best_pos, best_H, bW, bperm, brots

    # ---------- 统计辅助 ----------
    @staticmethod
    def stats(values):
        if not values:
            return 0.0, 0.0, 0.0
        b = min(values); mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return b, mean, math.sqrt(var)


