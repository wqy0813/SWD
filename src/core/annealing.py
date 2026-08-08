# 本文件实现基于 B*-Tree 的模拟退火优化器，负责模块旋转、节点交换、
# 节点移动等扰动搜索，并针对不同题目计算面积、线长和轮廓约束代价。
"""VLSI floorplanning solver modules split from the original spr backup."""


import math
import random
from typing import Dict, List, Optional, Tuple

from .bstar_tree import BTree, ContourPacker
from .models import FloorplanResult, Module, Net

class SimulatedAnnealing:
    """
    Simulated Annealing optimizer for B*-tree based floorplanning.

    Performs three types of moves:
    1. Rotate: rotate a random module by 90 degrees
    2. Swap: swap two random modules in the B*-tree
    3. Move: delete a module and re-insert it at a different position
    """
    _normalization_cache = {}

    def __init__(self, modules: List[Module], hard_indices: List[int],
                 nets: List[Net] = None,
                 terminal_positions: Dict[str, Tuple[float, float]] = None,
                 fixed_outline: Tuple[float, float] = None,
                 random_seed: int = 42):
        self.modules = modules
        self.hard_indices = hard_indices  # indices of hard blocks
        self.nets = nets or []
        self.terminal_positions = terminal_positions or {}
        self.fixed_outline = fixed_outline  # (width, height) or None

        self.packer = ContourPacker(modules)
        random.seed(random_seed)

        # SA parameters
        self.T_initial = 10000.0
        self.T_final = 0.01
        self.cooling_rate = 0.95
        self.max_iter_per_temp = 50
        self.max_total_iter = 5000

        # 中文说明：以下参数对应参考论文中的 Fast-SA 三阶段思想。
        # stage 1 用高温做随机探索，stage 2 做近似贪心搜索，
        # stage 3 重新升温后继续爬山搜索。
        self.fast_sa_initial_acceptance = 0.95
        self.fast_sa_c = 100.0
        self.fast_sa_stage1_ratio = 0.20
        self.fast_sa_stage2_ratio = 0.25
        self.fast_sa_stage2_cooling = 0.85

        # 中文说明：total_block_area 是固定数据集的理论面积下界。
        # area_norm 和 wirelength_norm 会在 run() 开始时通过随机采样估计，
        # 对应论文中 A_norm、W_norm 的归一化思想。
        self.total_block_area = sum(
            self.modules[i].area for i in self.hard_indices
        )
        self.area_norm = max(self.total_block_area, 1e-9)
        self.wirelength_norm = 1.0

        # 中文说明：统一代价函数的三个参数。
        # Cost = alpha * A/A_norm + beta * W/W_norm
        #        + (1 - alpha - beta) * (R - R*)^2
        self.alpha = 0.5
        self.beta = 0.0
        self.target_aspect_ratio = 1.0
        # 中文说明：问题一改为两阶段优化。
        # 阶段1严格只优化面积，完全对应题目“面积最小”主目标；
        # 阶段2在阶段1面积预算内，集中把长宽比推向 1。
        self.problem1_stage = 1
        self.problem1_area_budget = None
        self.problem1_stage1_shape_weight = 0.0
        self.problem1_stage2_area_penalty = 100.0
        # 中文说明：问题一最终仍然面积优先；这里的长宽比上限只是搜索护栏，
        # 防止纯面积目标把 B*-Tree 推入极端细长、反而面积更差的局部最优。
        self.problem1_stage1_aspect_ratio_limit = 3.0
        self.problem1_stage1_aspect_ratio_penalty = 0.05

        # Objective weights
        self.w_hpwl = 1.0
        self.w_area = 1.0
        self.w_ar = 1.0       # aspect ratio penalty
        self.w_overlap = 1000.0  # overlap penalty (for fixed outline)
        self.w_outline = 10000.0  # outline violation penalty

        # Track best
        self.best_tree = None
        self.best_rotations = None
        self.best_cost = float('inf')
        self.best_positions = None
        self.best_width = 0.0
        self.best_height = 0.0
        self.normalization_sample_cap = 80

        self._module_name_to_tree_idx = {
            self.modules[idx].name: tree_idx
            for tree_idx, idx in enumerate(self.hard_indices)
        }
        self._net_pin_refs = []
        self._net_weights = []
        for net in self.nets:
            refs = []
            for pin_name in net.pins:
                if pin_name in self._module_name_to_tree_idx:
                    refs.append((self._module_name_to_tree_idx[pin_name], None))
                elif pin_name in self.terminal_positions:
                    refs.append((None, self.terminal_positions[pin_name]))
            if refs:
                self._net_pin_refs.append(refs)
                self._net_weights.append(net.weight)

    def _normalization_cache_key(self, problem_type: int):
        """Build a dataset/objective key for reusable normalization samples."""
        hard_modules = tuple(
            (
                self.modules[idx].name,
                self.modules[idx].width,
                self.modules[idx].height,
            )
            for idx in self.hard_indices
        )
        nets = tuple(
            (tuple(net.pins), net.weight)
            for net in self.nets
        )
        terminals = tuple(sorted(self.terminal_positions.items()))
        outline = None
        if self.fixed_outline is not None:
            outline = (
                round(self.fixed_outline[0], 9),
                round(self.fixed_outline[1], 9),
            )
        return (
            problem_type,
            round(self.alpha, 9),
            round(self.beta, 9),
            round(self.target_aspect_ratio, 9),
            outline,
            hard_modules,
            nets,
            terminals,
        )

    def run(self, problem_type: int = 1, dead_space_ratio: float = 0.0,
            adaptive: bool = True, problem1_stage: int = 1,
            area_budget: Optional[float] = None,
            init_tree: Optional[BTree] = None,
            init_rotations: Optional[List[bool]] = None) -> FloorplanResult:
        """
        Run Simulated Annealing.

        Args:
            problem_type: 1 (area min), 2 (HPWL min), 3 (feasibility), 4 (non-rect)
            dead_space_ratio: for problem 2/3, the dead space ratio for outline
            adaptive: whether to use adaptive cooling schedule

        Returns:
            FloorplanResult with best found solution
        """
        # Initialize
        n = len(self.hard_indices)
        if n == 0:
            return FloorplanResult({}, 0, 0, 0, 0, 0, 0)

        self.problem1_stage = problem1_stage
        self.problem1_area_budget = area_budget

        # Initialize tree
        if init_tree is not None and init_rotations is not None:
            # 中文说明：两阶段优化时，第二阶段从第一阶段最优 B*-Tree 热启动。
            tree = init_tree.copy()
            rotations = init_rotations[:]
        else:
            tree = BTree(n)
            shuffled = list(range(n))
            random.shuffle(shuffled)
            tree.build_initial_tree(shuffled)

            # Initialize rotations (all not rotated)
            rotations = [False] * n

        # Compute fixed outline if needed
        if problem_type >= 2 and self.fixed_outline is None:
            total_area = sum(self.modules[i].area for i in self.hard_indices)
            side = math.sqrt(total_area * (1 + dead_space_ratio))
            self.fixed_outline = (side, side)

        # Initial packing
        self._apply_rotations(rotations)
        positions, width, height = self.packer.pack(tree)

        # 中文说明：正式计算 cost 前，先估计归一化常数 A_norm 和 W_norm。
        # 问题一固定 A_norm=模块总面积下界，不再用随机采样平均面积覆盖；
        # 问题二 beta>0 时才采样 HPWL。
        sample_count = min(max(20, n), self.normalization_sample_cap)
        self._estimate_normalization_constants(
            tree, rotations, problem_type, sample_count
        )
        current_cost = self._compute_cost(positions, width, height, problem_type)

        # Best tracking
        self.best_tree = tree.copy()
        self.best_rotations = rotations[:]
        self.best_cost = current_cost
        self.best_positions = dict(positions)
        self.best_width = width
        self.best_height = height

        if adaptive:
            # 中文说明：Fast-SA 预热采样。
            # 论文用若干随机邻域扰动估计平均上坡代价，
            # 再由 P = exp(-Delta_avg / T1) 反推出初始温度 T1。
            avg_uphill = self._estimate_uphill_cost(
                tree, rotations, current_cost, problem_type, sample_count
            )
            if avg_uphill > 0:
                T1 = max(
                    self.T_final * 10.0,
                    -avg_uphill / math.log(self.fast_sa_initial_acceptance)
                )
            else:
                T1 = self.T_initial

            # 中文说明：把总迭代次数分成论文对应的三个阶段。
            total_budget = max(3, self.max_total_iter)
            stage1_iters = max(1, int(total_budget * self.fast_sa_stage1_ratio))
            stage2_iters = max(1, int(total_budget * self.fast_sa_stage2_ratio))
            stage3_iters = max(1, total_budget - stage1_iters - stage2_iters)

            iteration = 0
            stuck_count = 0

            # 第一阶段：高温随机搜索，尽量扩大搜索范围。
            T = T1
            for _ in range(stage1_iters):
                tree, rotations, current_cost, stuck_count = self._attempt_move(
                    tree, rotations, current_cost, problem_type, T, stuck_count
                )
                iteration += 1

            # 第二阶段：伪贪心局部搜索，快速压低温度，主要接受优解。
            T = max(T1 / self.fast_sa_c, self.T_final * 10.0)
            for _ in range(stage2_iters):
                tree, rotations, current_cost, stuck_count = self._attempt_move(
                    tree, rotations, current_cost, problem_type, T, stuck_count
                )
                T = max(self.T_final, T * self.fast_sa_stage2_cooling)
                iteration += 1

            # 第三阶段：重新升温后继续爬山搜索，避免被第二阶段的局部最优锁死。
            T = max(T, T1 * 0.1)
            for _ in range(stage3_iters):
                tree, rotations, current_cost, stuck_count = self._attempt_move(
                    tree, rotations, current_cost, problem_type, T, stuck_count
                )
                # 中文说明：第三阶段不提前退出，保证预算真正用于重热后的局部细化。
                T = max(self.T_final, T * self.cooling_rate)
                iteration += 1
        else:
            # 中文说明：保留普通模拟退火作为对照实验，不使用 Fast-SA 三阶段。
            T = self.T_initial
            iteration = 0
            stuck_count = 0
            while T > self.T_final and iteration < self.max_total_iter:
                n_moves = min(self.max_iter_per_temp, n * 3)
                for _ in range(n_moves):
                    if iteration >= self.max_total_iter:
                        break
                    tree, rotations, current_cost, stuck_count = self._attempt_move(
                        tree, rotations, current_cost, problem_type, T, stuck_count
                    )
                    iteration += 1
                T *= self.cooling_rate

        # Restore best
        self._apply_rotations(self.best_rotations)

        # Compute final metrics
        area = self.best_width * self.best_height
        ar = max(self.best_width, self.best_height) / max(1e-9, min(self.best_width, self.best_height))

        # Build module name -> position mapping
        module_positions = {}
        for tree_idx, (x, y) in self.best_positions.items():
            mod = self.modules[self.hard_indices[tree_idx]]
            module_positions[mod.name] = (x, y, self.best_rotations[tree_idx])

        # Compute HPWL
        total_hpwl = 0.0
        if self.nets:
            total_hpwl = self._compute_hpwl_for_positions(self.best_positions)

        # Compute actual dead space ratio
        total_block_area = sum(self.modules[i].area for i in self.hard_indices)
        actual_dsr = (area - total_block_area) / max(1e-9, total_block_area)

        return FloorplanResult(
            module_positions=module_positions,
            outline_width=self.best_width,
            outline_height=self.best_height,
            area=area,
            aspect_ratio=ar,
            total_hpwl=total_hpwl,
            dead_space_ratio=actual_dsr
        )

    def _apply_rotations(self, rotations: List[bool]):
        """Apply rotation states to modules."""
        for i, idx in enumerate(self.hard_indices):
            self.modules[idx].rotated = rotations[i]

    def _perturb(self, tree: BTree, rotations: List[bool]):
        """生成一个 B*-Tree 邻域解：旋转、交换、移动模块或重组子树。"""
        n = len(self.hard_indices)
        move_type = random.choices(
            [1, 2, 3, 4, 5],
            # 中文说明：子树扰动代码已保留作对照实验；但问题一严格面积
            # 优先时，小样本测试发现子树扰动会提高长宽比却略增面积，
            # 因此默认仍采用论文中最基础的旋转/交换/移动三类扰动。
            weights=[0.30, 0.40, 0.30, 0.00, 0.00],
            k=1
        )[0]

        if move_type == 1:
            # 中文说明：旋转操作只改变模块方向，不改变 B*-Tree 拓扑。
            rotation_pos = random.randrange(n)
            rotations[rotation_pos] = not rotations[rotation_pos]
            return ('rotate', rotation_pos)
        elif move_type == 2:
            # 中文说明：交换操作改变两个节点对应的模块。
            i1, i2 = random.sample(range(n), 2)
            tree.swap_modules(i1, i2)
            return ('swap', (i1, i2))
        elif move_type == 3:
            # 中文说明：移动操作把一个节点删除后插入到另一节点附近。
            i1, i2 = random.sample(range(n), 2)
            old_tree = tree.copy()
            tree.delete_and_insert(i1, i2)
            return ('tree', old_tree)
        elif move_type == 4:
            # 中文说明：移动整棵子树，增强 B*-Tree 对大块局部结构的重组能力。
            i1, i2 = random.sample(range(n), 2)
            old_tree = tree.copy()
            tree.move_subtree(i1, i2)
            return ('tree', old_tree)

        # 中文说明：交换某个节点的左右子树，相当于局部改变“右侧/上方”关系。
        i = random.randrange(n)
        old_tree = tree.copy()
        tree.swap_children(i)
        return ('tree', old_tree)

    def _attempt_move(self, tree: BTree, rotations: List[bool],
                      current_cost: float, problem_type: int, temperature: float,
                      stuck_count: int):
        """尝试一次邻域扰动，并按模拟退火准则接受或恢复。"""
        old_rotations = rotations[:]

        undo_info = self._perturb(tree, rotations)
        self._apply_rotations(rotations)
        positions, width, height = self.packer.pack(tree)
        new_cost = self._compute_cost(positions, width, height, problem_type)
        delta = new_cost - current_cost

        accept_probability = math.exp(
            -delta / max(temperature, 1e-12)
        ) if delta > 0 else 1.0

        if delta <= 0 or random.random() < accept_probability:
            current_cost = new_cost
            if current_cost < self.best_cost:
                self.best_tree = tree.copy()
                self.best_rotations = rotations[:]
                self.best_cost = current_cost
                self.best_positions = dict(positions)
                self.best_width = width
                self.best_height = height
                stuck_count = 0
            else:
                stuck_count += 1
            return tree, rotations, current_cost, stuck_count

        # 中文说明：拒绝新解时恢复扰动前的树和旋转状态。
        self._apply_rotations(old_rotations)
        if undo_info[0] == 'swap':
            i1, i2 = undo_info[1]
            tree.swap_modules(i1, i2)
        elif undo_info[0] == 'tree':
            tree = undo_info[1]
        return tree, old_rotations, current_cost, stuck_count + 1

    def _estimate_uphill_cost(self, tree: BTree, rotations: List[bool],
                              current_cost: float, problem_type: int,
                              sample_count: int) -> float:
        """估计邻域扰动产生的平均上坡代价，用于计算 Fast-SA 初始温度。"""
        uphill_costs = []
        for _ in range(sample_count):
            trial_tree = tree.copy()
            trial_rotations = rotations[:]
            self._perturb(trial_tree, trial_rotations)
            self._apply_rotations(trial_rotations)
            positions, width, height = self.packer.pack(trial_tree)
            trial_cost = self._compute_cost(
                positions, width, height, problem_type
            )
            delta = trial_cost - current_cost
            if delta > 0:
                uphill_costs.append(delta)

        self._apply_rotations(rotations)
        if not uphill_costs:
            return 0.0
        return sum(uphill_costs) / len(uphill_costs)

    def _module_positions_for_hpwl(self, positions: Dict) -> Dict[str, Tuple[float, float]]:
        """把 B*-Tree 的内部下标坐标转换成模块名坐标，用于 HPWL 计算。"""
        mod_pos = {}
        for tree_idx, (x, y) in positions.items():
            mod = self.modules[self.hard_indices[tree_idx]]
            mod_pos[mod.name] = (x, y)
        return mod_pos

    def _compute_hpwl_for_positions(self, positions: Dict) -> float:
        """计算当前布局的 HPWL；没有线网时返回 0。"""
        if not self._net_pin_refs:
            return 0.0
        total_hpwl = 0.0
        modules = self.modules
        hard_indices = self.hard_indices
        center_x = [None] * len(hard_indices)
        center_y = [None] * len(hard_indices)
        for tree_idx, pos in positions.items():
            mod = modules[hard_indices[tree_idx]]
            center_x[tree_idx] = pos[0] + mod.w * 0.5
            center_y[tree_idx] = pos[1] + mod.h * 0.5

        for refs, weight in zip(self._net_pin_refs, self._net_weights):
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            found = False
            for tree_idx, terminal_xy in refs:
                if tree_idx is None:
                    px, py = terminal_xy
                else:
                    px = center_x[tree_idx]
                    if px is None:
                        continue
                    py = center_y[tree_idx]

                if px < min_x:
                    min_x = px
                if px > max_x:
                    max_x = px
                if py < min_y:
                    min_y = py
                if py > max_y:
                    max_y = py
                found = True

            if found:
                total_hpwl += ((max_x - min_x) + (max_y - min_y)) * weight

        return total_hpwl

    def _estimate_normalization_constants(self, tree: BTree, rotations: List[bool],
                                          problem_type: int, sample_count: int):
        """通过预热采样估计 A_norm 和 W_norm，对应论文中的归一化分母。"""
        if problem_type == 1:
            # 中文说明：问题一没有固定轮廓和连线，面积的天然归一化分母
            # 就是全部模块面积和，也就是理论面积下界。
            self.area_norm = max(self.total_block_area, 1e-9)
            self.wirelength_norm = 1.0
            return

        if problem_type != 2:
            self.area_norm = max(self.total_block_area, 1e-9)
            self.wirelength_norm = 1.0
            return

        cache_key = self._normalization_cache_key(problem_type)
        cached = SimulatedAnnealing._normalization_cache.get(cache_key)
        if cached is not None:
            self.area_norm, self.wirelength_norm = cached
            return

        areas = []
        wirelengths = []

        positions, width, height = self.packer.pack(tree)
        areas.append(width * height)
        if self.beta > 0:
            wirelengths.append(self._compute_hpwl_for_positions(positions))

        for _ in range(sample_count):
            trial_tree = tree.copy()
            trial_rotations = rotations[:]
            self._perturb(trial_tree, trial_rotations)
            self._apply_rotations(trial_rotations)
            trial_positions, trial_width, trial_height = self.packer.pack(trial_tree)
            areas.append(trial_width * trial_height)
            if self.beta > 0:
                wirelengths.append(self._compute_hpwl_for_positions(trial_positions))

        self._apply_rotations(rotations)

        if areas:
            self.area_norm = max(sum(areas) / len(areas), 1e-9)
        else:
            self.area_norm = max(self.total_block_area, 1e-9)

        if wirelengths and max(wirelengths) > 0:
            self.wirelength_norm = max(sum(wirelengths) / len(wirelengths), 1e-9)
        else:
            self.wirelength_norm = 1.0

        SimulatedAnnealing._normalization_cache[cache_key] = (
            self.area_norm,
            self.wirelength_norm,
        )

    def _compute_unified_floorplan_cost(self, positions: Dict,
                                        width: float, height: float) -> float:
        """统一归一化代价函数，问题一和问题二共用这一套公式。"""
        area = width * height
        aspect_ratio = max(width, height) / max(1e-9, min(width, height))

        cost_area = area / max(self.area_norm, 1e-9)
        if self.beta > 0:
            wirelength = self._compute_hpwl_for_positions(positions)
            cost_wirelength = wirelength / max(self.wirelength_norm, 1e-9)
        else:
            cost_wirelength = 0.0

        shape_weight = max(0.0, 1.0 - self.alpha - self.beta)
        cost_shape = (aspect_ratio - self.target_aspect_ratio) ** 2

        return (
            self.alpha * cost_area
            + self.beta * cost_wirelength
            + shape_weight * cost_shape
        )

    def _compute_problem1_cost(self, width: float, height: float) -> float:
        """问题一两阶段归一化目标函数。"""
        area = width * height
        aspect_ratio = max(width, height) / max(1e-9, min(width, height))
        area_cost = area / max(self.area_norm, 1e-9)
        shape_cost = (aspect_ratio - self.target_aspect_ratio) ** 2

        if self.problem1_stage == 1:
            # 中文说明：阶段1仍以面积为主；只有长宽比超过上限时才加护栏惩罚。
            if aspect_ratio <= self.problem1_stage1_aspect_ratio_limit:
                return area_cost
            overflow = aspect_ratio - self.problem1_stage1_aspect_ratio_limit
            return (
                area_cost
                + self.problem1_stage1_aspect_ratio_penalty * overflow ** 2
            )

        budget = self.problem1_area_budget or area
        over_budget_ratio = max(0.0, area - budget) / max(budget, 1e-9)
        # 中文说明：阶段2不是重新追求更大正方形，而是在阶段1面积预算内修正比例。
        return shape_cost + self.problem1_stage2_area_penalty * over_budget_ratio

    def _compute_cost(self, positions: Dict, width: float, height: float,
                      problem_type: int) -> float:
        """Compute the cost function based on problem type."""
        cost = 0.0

        if problem_type == 1:
            # 中文说明：问题一采用两阶段目标：
            # stage 1 面积主导；stage 2 在面积预算内优化长宽比。
            cost = self._compute_problem1_cost(width, height)

        elif problem_type == 2:
            # Minimize HPWL with fixed outline constraint
            # Penalize outline violations
            ow, oh = self.fixed_outline
            outline_penalty = max(0, width - ow) + max(0, height - oh)
            outline_penalty *= self.w_outline

            # 中文说明：问题二在统一公式基础上额外加入固定轮廓越界惩罚。
            cost = self._compute_unified_floorplan_cost(
                positions, width, height
            ) + outline_penalty

        elif problem_type == 3:
            # Feasibility check: only penalty for outline violation
            ow, oh = self.fixed_outline
            outline_penalty = max(0, width - ow) + max(0, height - oh)
            cost = outline_penalty

        elif problem_type == 4:
            # Same as problem 1 but with sub-block decomposition for non-rect modules
            area = width * height
            ar = max(width, height) / max(1e-9, min(width, height))
            cost = area * (1.0 + 0.1 * abs(ar - 1.0))

        return cost


# ============================================================
# PROBLEM SOLVERS
# ============================================================
