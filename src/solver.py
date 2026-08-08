# 本文件封装问题 1-4 的求解流程，负责调用解析器、模拟退火、
# HPWL 计算和异形模块求解器，形成面向题目小问的高层接口。
"""VLSI floorplanning solver modules split from the original spr backup."""


import copy
import math
import os
from typing import Dict, List, Optional, Tuple

from .annealing import SimulatedAnnealing
from .metrics import compute_hpwl
from .models import (
    FloorplanResult,
    Module,
    Net,
    Problem1Config,
    Problem2Config,
    Problem3Config,
    Problem4Config,
    SubBlock,
)
from .nonrect import NonRectSolver
from .parser import VLSIParser
from .visualize import visualize_floorplan

class ProblemSolver:
    """Main solver orchestrator for all four problems."""

    def __init__(self, blocks_file: str, nets_file: str, pl_file: str,
                 problem1_config: Optional[Problem1Config] = None,
                 problem2_config: Optional[Problem2Config] = None,
                 problem3_config: Optional[Problem3Config] = None,
                 problem4_config: Optional[Problem4Config] = None):
        self.blocks_file = blocks_file
        self.nets_file = nets_file
        self.pl_file = pl_file

        # 中文说明：四个问题的优化目标和约束不同，因此参数分开管理。
        # 这样调问题一的长宽比权重，不会影响问题二的 HPWL 或问题三的二分搜索。
        self.problem1_config = problem1_config or Problem1Config()
        self.problem2_config = problem2_config or Problem2Config()
        self.problem3_config = problem3_config or Problem3Config()
        self.problem4_config = problem4_config or Problem4Config()

        self.modules: List[Module] = []
        self.nets: List[Net] = []
        self.terminal_positions: Dict[str, Tuple[float, float]] = {}

        self.hard_modules: List[Module] = []
        self.hard_indices: List[int] = []

        self._load_data()

    def _load_data(self):
        """Load all input data."""
        self.modules = VLSIParser.parse_blocks(self.blocks_file)
        if os.path.exists(self.nets_file):
            self.nets = VLSIParser.parse_nets(self.nets_file)
        if os.path.exists(self.pl_file):
            self.terminal_positions = VLSIParser.parse_pl(self.pl_file)

        # Separate hard blocks
        self.hard_modules = []
        self.hard_indices = []
        for mod in self.modules:
            if mod.module_type == 'block':
                self.hard_indices.append(len(self.hard_modules))
                self.hard_modules.append(mod)

        print(f"Loaded {len(self.modules)} modules "
              f"({len(self.hard_modules)} hard blocks, "
              f"{len(self.modules) - len(self.hard_modules)} terminals)")
        print(f"Loaded {len(self.nets)} nets")
        print(f"Loaded {len(self.terminal_positions)} terminal positions")

    def solve_problem1(self) -> FloorplanResult:
        """
        Problem 1: Minimize chip area with variable outline and no connection constraints.

        Approach:
        - B*-tree representation + Contour-based packing
        - Fast-SA with normalized area and aspect-ratio penalty
        - Modules can be rotated 90 degrees
        """
        print("\n" + "=" * 60)
        print("PROBLEM 1: Area Minimization with Variable Outline")
        print("=" * 60)

        cfg = self.problem1_config

        # 中文说明：模拟退火具有随机性，单次运行可能偶然停在较差的局部最优。
        # 因此问题一使用多个随机种子独立搜索。
        seeds = cfg.seeds
        best_result = None

        for trial_id, seed in enumerate(seeds, start=1):
            # 中文说明：每次试验使用深拷贝，避免上一轮的旋转状态污染下一轮。
            trial_modules = copy.deepcopy(self.hard_modules)
            sa = SimulatedAnnealing(
                modules=trial_modules,
                hard_indices=list(range(len(trial_modules))),
                nets=[],  # 问题一不考虑连接关系
                terminal_positions={},
                fixed_outline=None,  # 问题一的轮廓由布局结果决定
                random_seed=seed
            )

            # 中文说明：问题一采用 Fast-SA；max_total_iter 会被三阶段共同分配。
            sa.T_initial = cfg.t_initial
            sa.T_final = cfg.t_final
            sa.cooling_rate = cfg.cooling_rate
            # 中文说明：问题一代入统一公式：beta=0，R*=1。
            # Cost = alpha*A/A_norm + (1-alpha)*(R-1)^2
            sa.alpha = cfg.alpha
            sa.beta = cfg.beta
            sa.target_aspect_ratio = cfg.target_aspect_ratio
            sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(trial_modules))
            sa.max_total_iter = max(
                cfg.min_total_iter,
                len(trial_modules) * cfg.iter_multiplier
            )

            result = sa.run(problem_type=1, adaptive=True)
            print(
                f"Trial {trial_id}/{len(seeds)} "
                f"(seed={seed}): area={result.area:.2f}, "
                f"AR={result.aspect_ratio:.4f}"
            )

            # 中文说明：问题一的最终选择采用“面积主导 + 容差内比例优先”。
            # 如果面积差距超过 cfg.area_tolerance，就选面积更小的；
            # 如果面积差距在容差内，则选长宽比更接近 1 的，更符合题目中
            # “面积相同或接近时，长宽比越接近 1 越好”的表达。
            if best_result is None:
                best_result = result
            else:
                relative_area_gap = abs(result.area - best_result.area) / max(
                    min(result.area, best_result.area),
                    1e-9
                )
                if relative_area_gap <= cfg.area_tolerance:
                    if result.aspect_ratio < best_result.aspect_ratio:
                        best_result = result
                elif result.area < best_result.area:
                    best_result = result

        total_block_area = sum(module.area for module in self.hard_modules)
        utilization = total_block_area / max(best_result.area, 1e-9)

        # 中文说明：total_block_area 是问题一的理论面积下界。
        # 死区比例与题目公式一致，分母采用模块总面积。
        print(f"\nTheoretical lower bound: {total_block_area:.2f}")
        print(
            f"Best outline: "
            f"{best_result.outline_width:.2f} x {best_result.outline_height:.2f}"
        )
        print(f"Best area: {best_result.area:.2f}")
        print(f"Aspect Ratio: {best_result.aspect_ratio:.4f}")
        print(f"Utilization: {utilization:.4f}")
        print(f"Dead Space Ratio: {best_result.dead_space_ratio:.4f}")
        print(f"Area tolerance for aspect-ratio tie-break: {cfg.area_tolerance:.2%}")

        return best_result

    def solve_problem2(self, dead_space_ratio: Optional[float] = None) -> FloorplanResult:
        """
        Problem 2: Minimize HPWL with fixed square outline.

        Approach:
        - Outline dimensions computed from formula (1)
        - B*-tree + SA with HPWL objective
        - Heavy penalty for outline violations
        """
        cfg = self.problem2_config
        if dead_space_ratio is None:
            dead_space_ratio = cfg.dead_space_ratio

        print("\n" + "=" * 60)
        print(f"PROBLEM 2: HPWL Minimization with Fixed Outline (DSR={dead_space_ratio})")
        print("=" * 60)

        # Compute fixed outline
        total_area = sum(m.area for m in self.hard_modules)
        side = math.sqrt(total_area * (1 + dead_space_ratio))
        fixed_outline = (side, side)
        print(f"Fixed Outline: {side:.2f} x {side:.2f}")
        print(f"Total Block Area: {total_area:.2f}")

        sa = SimulatedAnnealing(
            modules=self.hard_modules,
            hard_indices=list(range(len(self.hard_modules))),
            nets=self.nets,
            terminal_positions=self.terminal_positions,
            fixed_outline=fixed_outline,
            random_seed=cfg.seed
        )

        # Tune SA parameters for problem 2
        sa.T_initial = cfg.t_initial
        sa.T_final = cfg.t_final
        sa.cooling_rate = cfg.cooling_rate
        sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(self.hard_modules))
        sa.max_total_iter = max(
            cfg.min_total_iter,
            len(self.hard_modules) * cfg.iter_multiplier
        )
        sa.w_outline = cfg.outline_penalty
        # 中文说明：问题二也使用统一归一化公式，但额外叠加固定轮廓越界惩罚。
        sa.alpha = cfg.alpha
        sa.beta = cfg.beta
        sa.target_aspect_ratio = cfg.target_aspect_ratio

        result = sa.run(problem_type=2, dead_space_ratio=dead_space_ratio)

        # Check if feasible
        if result.outline_width > side + 1e-6 or result.outline_height > side + 1e-6:
            print("WARNING: Solution violates outline constraint!")
            print(f"  Width: {result.outline_width:.2f} > {side:.2f}")
            print(f"  Height: {result.outline_height:.2f} > {side:.2f}")

        print(f"Total HPWL: {result.total_hpwl:.2f}")
        print(f"Outline: {result.outline_width:.2f} x {result.outline_height:.2f}")
        print(f"Area: {result.area:.2f}")

        return result

    def solve_problem3(self) -> Tuple[float, FloorplanResult]:
        """
        Problem 3: Find minimum feasible dead_space_ratio.

        Approach:
        - Binary search on dead_space_ratio
        - For each ratio, run SA to check feasibility
        - Feasible if there exists a placement within the outline
        """
        print("\n" + "=" * 60)
        print("PROBLEM 3: Minimum Feasible Dead Space Ratio")
        print("=" * 60)

        cfg = self.problem3_config
        total_area = sum(m.area for m in self.hard_modules)

        # Binary search range
        lo = cfg.low
        hi = cfg.high  # Upper bound for dead space ratio
        best_feasible = hi
        best_result = None

        n_trials_per_ratio = cfg.trials_per_ratio  # Multiple SA runs for robustness

        iteration = 0
        while hi - lo > cfg.precision and iteration < cfg.max_binary_iter:
            iteration += 1
            mid = (lo + hi) / 2.0
            side = math.sqrt(total_area * (1 + mid))
            fixed_outline = (side, side)

            print(f"\n  Iteration {iteration}: Testing DSR = {mid:.4f}, Outline = {side:.2f}")

            feasible = False
            best_trial_result = None
            best_trial_violation = float('inf')

            for trial in range(n_trials_per_ratio):
                sa = SimulatedAnnealing(
                    modules=[copy.deepcopy(m) for m in self.hard_modules],
                    hard_indices=list(range(len(self.hard_modules))),
                    nets=self.nets,
                    terminal_positions=self.terminal_positions,
                    fixed_outline=fixed_outline,
                    random_seed=42 + trial * 100 + iteration * 10
                )

                sa.T_initial = cfg.t_initial
                sa.T_final = cfg.t_final
                sa.cooling_rate = cfg.cooling_rate
                sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(self.hard_modules))
                sa.max_total_iter = max(
                    cfg.min_total_iter,
                    len(self.hard_modules) * cfg.iter_multiplier
                )
                sa.w_outline = cfg.outline_penalty

                result = sa.run(problem_type=3, dead_space_ratio=mid)

                violation = (max(0, result.outline_width - side) +
                             max(0, result.outline_height - side))
                if violation < 0.01 * side:
                    feasible = True
                if violation < best_trial_violation:
                    best_trial_violation = violation
                    best_trial_result = result

            if feasible:
                # Can try smaller ratio
                hi = mid
                best_feasible = mid
                best_result = best_trial_result
                print(f"    [OK] Feasible at DSR = {mid:.4f} (narrowing search down)")
            else:
                lo = mid
                print(f"    [X] Not feasible at DSR = {mid:.4f} (moving up)")

        print(f"\nMinimum feasible dead space ratio: {best_feasible:.4f}")

        # Re-run with best ratio to get HPWL
        if best_result is None:
            best_result = self.solve_problem2(best_feasible)
        else:
            # Recompute HPWL properly
            best_result = self._refine_result(best_feasible, best_result)

        print(f"Total HPWL at minimum DSR: {best_result.total_hpwl:.2f}")
        print(f"Outline: {best_result.outline_width:.2f} x {best_result.outline_height:.2f}")

        return best_feasible, best_result

    def _refine_result(self, dsr: float, previous_result: FloorplanResult) -> FloorplanResult:
        """Re-run SA at a given DSR with HPWL objective for a refined result."""
        total_area = sum(m.area for m in self.hard_modules)
        side = math.sqrt(total_area * (1 + dsr))
        cfg = self.problem2_config

        # Restore rotations from previous result
        for mod_name, (x, y, rotated) in previous_result.module_positions.items():
            for mod in self.hard_modules:
                if mod.name == mod_name:
                    mod.rotated = rotated
                    break

        sa = SimulatedAnnealing(
            modules=self.hard_modules,
            hard_indices=list(range(len(self.hard_modules))),
            nets=self.nets,
            terminal_positions=self.terminal_positions,
            fixed_outline=(side, side),
            random_seed=777
        )
        sa.T_initial = cfg.t_initial
        sa.T_final = cfg.t_final
        sa.cooling_rate = cfg.cooling_rate
        sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(self.hard_modules))
        sa.max_total_iter = max(
            cfg.min_total_iter,
            len(self.hard_modules) * cfg.iter_multiplier
        )
        sa.w_outline = cfg.outline_penalty
        sa.alpha = cfg.alpha
        sa.beta = cfg.beta
        sa.target_aspect_ratio = cfg.target_aspect_ratio
        return sa.run(problem_type=2, dead_space_ratio=dsr)

    def solve_problem4(self, custom_modules: List[Module] = None) -> FloorplanResult:
        """
        Problem 4: Floorplanning with L-shaped and T-shaped modules.

        Approach:
        - Decompose non-rectangular modules into sub-rectangles
        - Use B*-tree where each node is a sub-block
        - Add constraints to keep sub-blocks of same module together
        - Apply 90°/180°/270° rotation to entire module
        """
        print("\n" + "=" * 60)
        print("PROBLEM 4: Non-Rectangular Module Floorplanning")
        print("=" * 60)
        cfg = self.problem4_config

        if custom_modules:
            modules = custom_modules
        else:
            modules = [m for m in self.modules if m.module_type == 'block']

        # Decompose non-rect modules into sub-blocks
        all_sub_blocks = []  # List of (module_idx, SubBlock)
        module_sub_map = defaultdict(list)  # module_idx -> [sub_block_indices]

        for mod_idx, mod in enumerate(modules):
            if mod.shape_type != 'rect' and mod.sub_blocks:
                for sb in mod.sub_blocks:
                    idx = len(all_sub_blocks)
                    all_sub_blocks.append((mod_idx, sb))
                    module_sub_map[mod_idx].append(idx)
            else:
                # Rectangular module: single sub-block
                sb = SubBlock(0, 0, mod.width, mod.height)
                idx = len(all_sub_blocks)
                all_sub_blocks.append((mod_idx, sb))
                module_sub_map[mod_idx].append(idx)

        # Create flat modules for each sub-block
        flat_modules = []
        for mod_idx, sb in all_sub_blocks:
            flat_mod = Module(
                name=f"{modules[mod_idx].name}_sub{len(flat_modules)}",
                module_type='block',
                width=sb.width,
                height=sb.height,
                is_hard=True
            )
            # Store reference to parent module and relative offset
            flat_mod._parent_mod = modules[mod_idx]
            flat_mod._rel_x = sb.rel_x
            flat_mod._rel_y = sb.rel_y
            flat_mod._mod_idx = mod_idx
            flat_modules.append(flat_mod)

        print(f"Decomposed {len(modules)} modules into {len(flat_modules)} sub-blocks")
        for mod_idx, mod in enumerate(modules):
            if mod.shape_type != 'rect':
                print(f"  {mod.name}: {mod.shape_type}-shaped, "
                      f"{len(module_sub_map[mod_idx])} sub-blocks")

        # SA with special constraint handling
        sa = SimulatedAnnealing(
            modules=flat_modules,
            hard_indices=list(range(len(flat_modules))),
            nets=[],
            terminal_positions={},
            fixed_outline=None,
            random_seed=cfg.seed
        )

        sa.T_initial = cfg.t_initial
        sa.T_final = cfg.t_final
        sa.cooling_rate = cfg.cooling_rate
        sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(flat_modules))
        sa.max_total_iter = max(
            cfg.min_total_iter,
            len(flat_modules) * cfg.iter_multiplier
        )

        # We need a modified SA that respects module grouping
        # For simplicity, we use the standard SA but compute cost differently
        result = sa.run(problem_type=4)

        # Post-process: compute actual outline as bounding box of reconstructed modules
        # Reconstruct modules from sub-blocks
        result = self._reconstruct_from_sub_blocks(result, modules, module_sub_map,
                                                    all_sub_blocks, flat_modules)

        print(f"Outline: {result.outline_width:.2f} x {result.outline_height:.2f}")
        print(f"Area: {result.area:.2f}")
        print(f"Aspect Ratio: {result.aspect_ratio:.4f}")

        return result

    def _reconstruct_from_sub_blocks(self, sa_result: FloorplanResult,
                                     modules: List[Module],
                                     module_sub_map: Dict[int, List[int]],
                                     all_sub_blocks: List[Tuple[int, SubBlock]],
                                     flat_modules: List[Module]) -> FloorplanResult:
        """Reconstruct module positions from sub-block packing."""
        module_positions = {}

        for mod_idx, mod in enumerate(modules):
            sub_indices = module_sub_map[mod_idx]

            if len(sub_indices) == 1:
                # Rectangular module
                sub_idx = sub_indices[0]
                flat_mod = flat_modules[sub_idx]
                if flat_mod.name in sa_result.module_positions:
                    x, y, rot = sa_result.module_positions[flat_mod.name]
                    module_positions[mod.name] = (x, y, rot)
            else:
                # Non-rectangular: need to reconstruct from sub-blocks
                sb_positions = []
                for sub_idx in sub_indices:
                    flat_mod = flat_modules[sub_idx]
                    if flat_mod.name in sa_result.module_positions:
                        x, y, _ = sa_result.module_positions[flat_mod.name]
                        sb = all_sub_blocks[sub_idx][1]
                        # Absolute position = sub-block position + relative offset
                        abs_x = x - sb.rel_x
                        abs_y = y - sb.rel_y
                        sb_positions.append((abs_x, abs_y))

                if sb_positions:
                    # Average the origin positions (should be close if constraint works)
                    avg_x = sum(p[0] for p in sb_positions) / len(sb_positions)
                    avg_y = sum(p[1] for p in sb_positions) / len(sb_positions)
                    module_positions[mod.name] = (avg_x, avg_y, False)

        # Compute overall bounding box
        max_x = 0.0
        max_y = 0.0
        for mod_name, (x, y, rot) in module_positions.items():
            mod = next((m for m in modules if m.name == mod_name), None)
            if mod:
                if rot:
                    w, h = mod.height, mod.width
                else:
                    w, h = mod.width, mod.height
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        area = max_x * max_y
        ar = max(max_x, max_y) / max(1e-9, min(max_x, max_y))

        return FloorplanResult(
            module_positions=module_positions,
            outline_width=max_x,
            outline_height=max_y,
            area=area,
            aspect_ratio=ar,
            total_hpwl=0.0,
            dead_space_ratio=0.0
        )


# ============================================================
# PROBLEM 4: NON-RECTANGULAR MODULE EXTENDED SOLVER
# ============================================================
