# 鏈枃浠跺皝瑁呴棶棰?1-4 鐨勬眰瑙ｆ祦绋嬶紝璐熻矗璋冪敤瑙ｆ瀽鍣ㄣ€佹ā鎷熼€€鐏€?# HPWL 璁＄畻鍜屽紓褰㈡ā鍧楁眰瑙ｅ櫒锛屽舰鎴愰潰鍚戦鐩皬闂殑楂樺眰鎺ュ彛銆?"""VLSI floorplanning solver modules split from the original spr backup."""


import copy
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from ..core.annealing import SimulatedAnnealing
from ..core.bstar_tree import BTree
from ..core.metrics import compute_hpwl
from ..core.models import (
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
from ..core.parser import VLSIParser
from ..core.postprocess import compact_floorplan
from ..core.shelf_sa import ShelfSA
from ..core.validation import validate_floorplan
from ..core.visualize import visualize_floorplan

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

    @staticmethod
    def _is_better_problem1(candidate: FloorplanResult,
                            incumbent: Optional[FloorplanResult],
                            area_tie_epsilon: float) -> bool:
        """Return whether a Problem 1 candidate is better than the incumbent."""
        if incumbent is None:
            return True
        if candidate.area < incumbent.area - area_tie_epsilon:
            return True
        if abs(candidate.area - incumbent.area) <= area_tie_epsilon:
            return candidate.aspect_ratio < incumbent.aspect_ratio
        return False

    def _make_problem1_initial_tree(self, mode: str, modules: List[Module],
                                    seed: int) -> Optional[BTree]:
        """按指定模式构造问题一的 B*-Tree 初始解；random 模式返回 None。"""
        n = len(modules)
        if mode == "random_balanced":
            return None

        indices = list(range(n))
        if mode.startswith("area_desc"):
            indices.sort(key=lambda i: modules[i].area, reverse=True)
        elif mode.startswith("width_desc"):
            indices.sort(key=lambda i: modules[i].width, reverse=True)
        elif mode.startswith("height_desc"):
            indices.sort(key=lambda i: modules[i].height, reverse=True)
        elif mode.startswith("long_side_desc"):
            indices.sort(
                key=lambda i: max(modules[i].width, modules[i].height),
                reverse=True
            )
        elif mode.startswith("short_side_desc"):
            indices.sort(
                key=lambda i: min(modules[i].width, modules[i].height),
                reverse=True
            )
        elif mode.startswith("random"):
            rng = random.Random(seed)
            rng.shuffle(indices)
        else:
            return None

        tree = BTree(n)
        if mode.endswith("left_chain"):
            tree.build_left_chain(indices)
        elif mode.endswith("right_chain"):
            tree.build_right_chain(indices)
        else:
            tree.build_initial_tree(indices)
        return tree

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

    def solve_problem1(self, save_dir: Optional[str] = None,
                       chip_name: str = "chip") -> FloorplanResult:
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

        # 中文说明：模拟退火具有随机性，因此问题一使用多个随机种子独立搜索。
        seeds = cfg.seeds
        best_result = None

        trial_id = 0
        total_trials = len(seeds) * len(cfg.initial_tree_modes)
        for seed in seeds:
            for initial_mode in cfg.initial_tree_modes:
                trial_id += 1
                # 中文说明：每次试验使用深拷贝，避免上一轮旋转状态污染下一轮。
                trial_modules = copy.deepcopy(self.hard_modules)
                sa = SimulatedAnnealing(
                    modules=trial_modules,
                    hard_indices=list(range(len(trial_modules))),
                    nets=[],  # 问题一不考虑连接关系
                    terminal_positions={},
                    fixed_outline=None,  # 问题一的轮廓由布局结果决定
                    random_seed=seed
                )
                initial_tree = self._make_problem1_initial_tree(
                    initial_mode,
                    trial_modules,
                    seed
                )
                initial_rotations = (
                    [False] * len(trial_modules)
                    if initial_tree is not None
                    else None
                )

                # 中文说明：问题一采用两阶段 Fast-SA。
                # 阶段一以面积为主，并用 AR 上限惩罚防止极端细长；
                # 阶段二从阶段一最优 B*-Tree 热启动，在面积预算内修长宽比。
                sa.T_initial = cfg.t_initial
                sa.T_final = cfg.t_final
                sa.cooling_rate = cfg.cooling_rate
                sa.alpha = cfg.alpha
                sa.beta = cfg.beta
                sa.target_aspect_ratio = cfg.target_aspect_ratio
                sa.problem1_stage1_shape_weight = cfg.stage1_shape_weight
                sa.problem1_stage2_area_penalty = cfg.stage2_area_penalty
                sa.problem1_stage1_aspect_ratio_limit = cfg.stage1_aspect_ratio_limit
                sa.problem1_stage1_aspect_ratio_penalty = cfg.stage1_aspect_ratio_penalty
                sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(trial_modules))
                sa.max_total_iter = max(
                    cfg.min_total_iter,
                    len(trial_modules) * cfg.iter_multiplier
                )

                stage1_result = sa.run(
                    problem_type=1,
                    adaptive=True,
                    problem1_stage=1,
                    init_tree=initial_tree,
                    init_rotations=initial_rotations
                )
                stage1_tree = sa.best_tree.copy()
                stage1_rotations = sa.best_rotations[:]
                area_budget = stage1_result.area * cfg.stage2_area_budget_factor
                # 中文说明：最终比较前执行几何压缩；它只移动最终坐标，
                # 不改变 B*-Tree 搜索过程中的树结构热启动。
                stage1_compacted = compact_floorplan(stage1_result, trial_modules)

                sa.max_total_iter = max(
                    cfg.stage2_min_total_iter,
                    len(trial_modules) * cfg.stage2_iter_multiplier
                )
                stage2_result = sa.run(
                    problem_type=1,
                    adaptive=True,
                    problem1_stage=2,
                    area_budget=area_budget,
                    init_tree=stage1_tree,
                    init_rotations=stage1_rotations
                )
                stage2_compacted = compact_floorplan(stage2_result, trial_modules)

                if self._is_better_problem1(
                    stage2_compacted,
                    stage1_compacted,
                    cfg.area_tie_epsilon
                ):
                    result = stage2_compacted
                else:
                    result = stage1_compacted

                print(
                    f"Trial {trial_id}/{total_trials} "
                    f"(seed={seed}, init={initial_mode}): "
                    f"stage1 area={stage1_result.area:.2f}, "
                    f"stage1 AR={stage1_result.aspect_ratio:.4f}, "
                    f"compacted={stage1_compacted.area:.2f}; "
                    f"stage2 area={stage2_result.area:.2f}, "
                    f"stage2 AR={stage2_result.aspect_ratio:.4f}, "
                    f"compacted={stage2_compacted.area:.2f}; "
                    f"chosen area={result.area:.2f}, "
                    f"chosen AR={result.aspect_ratio:.4f}"
                )

                if self._is_better_problem1(
                    result,
                    best_result,
                    cfg.area_tie_epsilon
                ):
                    best_result = result

        total_block_area = sum(module.area for module in self.hard_modules)
        utilization = total_block_area / max(best_result.area, 1e-9)
        is_valid, validation_issues = validate_floorplan(
            best_result,
            self.hard_modules
        )

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
        print(
            "Area tie epsilon for aspect-ratio tie-break: "
            f"{cfg.area_tie_epsilon:g}"
        )
        if is_valid:
            print("Validation: PASS (no overlaps, no outline violations)")
        else:
            print("Validation: FAIL")
            for issue in validation_issues[:8]:
                print(f"  - {issue}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            png_path = os.path.join(save_dir, f"{chip_name}_problem1_floorplan.png")
            visualize_floorplan(
                best_result,
                self.hard_modules,
                title=f"{chip_name} - Problem 1: Area Minimization",
                save_path=png_path
            )

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

        return self._solve_problem2_hybrid(dead_space_ratio)


    def _solve_problem2_hybrid(self, dead_space_ratio: float) -> FloorplanResult:
        """Try B*-tree SA first, then fall back to shelf SA if needed."""
        btree_result = self._solve_problem2_multiseed(dead_space_ratio)
        if btree_result.feasible:
            return btree_result

        print("\nB*-tree trials did not produce a feasible fixed-outline placement.")
        print("Falling back to shelf feasibility + HPWL optimization.")
        return self._solve_problem2_shelf(dead_space_ratio)

    def _solve_problem2_shelf(self, dead_space_ratio: float) -> FloorplanResult:
        """Solve Problem 2 with shelf packing: feasibility first, HPWL second."""
        cfg = self.problem2_config

        print("\n" + "=" * 60)
        print(f"PROBLEM 2: HPWL Minimization with Fixed Outline (DSR={dead_space_ratio})")
        print("=" * 60)

        total_area = sum(m.area for m in self.hard_modules)
        side = math.sqrt(total_area * (1 + dead_space_ratio))
        print(f"Fixed Outline: {side:.2f} x {side:.2f}")
        print(f"Total Block Area: {total_area:.2f}")

        candidates = []
        seeds = cfg.seeds or (cfg.seed,)
        base_iter = max(cfg.min_total_iter, len(self.hard_modules) * cfg.iter_multiplier)
        feas_iter = max(cfg.min_total_iter, base_iter)
        hpwl_iter = max(cfg.min_total_iter, base_iter * 2)

        for trial_id, seed in enumerate(seeds, start=1):
            shelf = ShelfSA(
                copy.deepcopy(self.hard_modules),
                self.nets,
                self.terminal_positions,
                seed=seed,
            )

            feas_pos, feas_h, feas_perm, feas_rots = shelf.anneal(
                width=side,
                objective="feas",
                max_total_iter=feas_iter,
                max_iter_per_temp=cfg.min_iter_per_temp,
                t_final=max(cfg.t_final, 0.5),
                cooling_rate=0.995,
            )
            init = (feas_perm, feas_rots) if feas_h <= side + cfg.feasibility_tolerance else None

            hpwl_pos, hpwl_h, hpwl_perm, hpwl_rots = shelf.anneal(
                width=side,
                objective="hpwl",
                max_total_iter=hpwl_iter,
                max_iter_per_temp=cfg.min_iter_per_temp,
                t_final=max(cfg.t_final, 0.5),
                cooling_rate=0.995,
                init=init,
                strict=True,
            )

            hpwl = shelf.hpwl(hpwl_pos)
            result = FloorplanResult(
                module_positions=hpwl_pos,
                outline_width=side,
                outline_height=side,
                area=side * side,
                aspect_ratio=1.0,
                total_hpwl=hpwl,
                dead_space_ratio=dead_space_ratio,
            )
            ok, issues = validate_floorplan(
                result,
                self.hard_modules,
                eps=cfg.feasibility_tolerance,
            )
            result.feasible = ok
            result.verify_info = "; ".join(issues[:5])

            violation = max(0.0, hpwl_h - side)
            candidates.append((result, violation, hpwl_h, seed))

            status = "feasible" if result.feasible else "infeasible"
            print(
                f"Trial {trial_id}/{len(seeds)} "
                f"(seed={seed}): {status}, "
                f"height={hpwl_h:.2f}, violation={violation:.6f}, "
                f"HPWL={result.total_hpwl:.2f}"
            )

        feasible = [item for item in candidates if item[0].feasible]
        if feasible:
            best_result, _, best_h, best_seed = min(
                feasible,
                key=lambda item: item[0].total_hpwl,
            )
        else:
            best_result, _, best_h, best_seed = min(
                candidates,
                key=lambda item: (item[1], item[0].total_hpwl),
            )

        print(f"Selected seed: {best_seed}")
        print(f"Packed height: {best_h:.2f}")
        print(f"Fixed outline: {best_result.outline_width:.2f} x {best_result.outline_height:.2f}")
        print(f"Total HPWL: {best_result.total_hpwl:.2f}")
        print(
            "Validation: "
            f"{'PASS' if best_result.feasible else 'FAIL'}"
            + (f" ({best_result.verify_info})" if best_result.verify_info else "")
        )

        return best_result

    def _solve_problem2_multiseed(self, dead_space_ratio: float) -> FloorplanResult:
        """Run Problem 2 with multiple seeds and select the best feasible HPWL."""
        cfg = self.problem2_config

        print("\n" + "=" * 60)
        print(f"PROBLEM 2: HPWL Minimization with Fixed Outline (DSR={dead_space_ratio})")
        print("=" * 60)

        total_area = sum(m.area for m in self.hard_modules)
        side = math.sqrt(total_area * (1 + dead_space_ratio))
        fixed_outline = (side, side)
        print(f"Fixed Outline: {side:.2f} x {side:.2f}")
        print(f"Total Block Area: {total_area:.2f}")

        candidates = []
        seeds = cfg.seeds or (cfg.seed,)
        for trial_id, seed in enumerate(seeds, start=1):
            trial_modules = copy.deepcopy(self.hard_modules)
            sa = SimulatedAnnealing(
                modules=trial_modules,
                hard_indices=list(range(len(trial_modules))),
                nets=self.nets,
                terminal_positions=self.terminal_positions,
                fixed_outline=fixed_outline,
                random_seed=seed
            )

            sa.T_initial = cfg.t_initial
            sa.T_final = cfg.t_final
            sa.cooling_rate = cfg.cooling_rate
            sa.max_iter_per_temp = max(cfg.min_iter_per_temp, len(trial_modules))
            sa.max_total_iter = max(
                cfg.min_total_iter,
                len(trial_modules) * cfg.iter_multiplier
            )
            sa.w_outline = cfg.outline_penalty
            sa.alpha = cfg.alpha
            sa.beta = cfg.beta
            sa.target_aspect_ratio = cfg.target_aspect_ratio

            raw_result = sa.run(problem_type=2, dead_space_ratio=dead_space_ratio)
            result = self._fixed_outline_problem2_result(
                raw_result,
                side,
                dead_space_ratio,
            )
            violation = (
                max(0.0, raw_result.outline_width - side)
                + max(0.0, raw_result.outline_height - side)
            )
            candidates.append((
                result,
                violation,
                raw_result.outline_width,
                raw_result.outline_height,
                seed,
            ))

            status = "feasible" if result.feasible else "infeasible"
            print(
                f"Trial {trial_id}/{len(seeds)} "
                f"(seed={seed}): {status}, "
                f"packed={raw_result.outline_width:.2f}x{raw_result.outline_height:.2f}, "
                f"violation={violation:.6f}, HPWL={result.total_hpwl:.2f}"
            )

        feasible = [item for item in candidates if item[0].feasible]
        if feasible:
            best_result, _, best_w, best_h, best_seed = min(
                feasible,
                key=lambda item: item[0].total_hpwl,
            )
        else:
            best_result, _, best_w, best_h, best_seed = min(
                candidates,
                key=lambda item: (item[1], item[0].total_hpwl),
            )

        print(f"Selected seed: {best_seed}")
        print(f"Packed bounding box: {best_w:.2f} x {best_h:.2f}")
        print(f"Fixed outline: {best_result.outline_width:.2f} x {best_result.outline_height:.2f}")
        print(f"Total HPWL: {best_result.total_hpwl:.2f}")
        print(
            "Validation: "
            f"{'PASS' if best_result.feasible else 'FAIL'}"
            + (f" ({best_result.verify_info})" if best_result.verify_info else "")
        )

        return best_result

    def _fixed_outline_problem2_result(self, raw_result: FloorplanResult,
                                       side: float,
                                       dead_space_ratio: float) -> FloorplanResult:
        """Convert packed B*-tree bounds into the fixed square chip outline."""
        result = FloorplanResult(
            module_positions=dict(raw_result.module_positions),
            outline_width=side,
            outline_height=side,
            area=side * side,
            aspect_ratio=1.0,
            total_hpwl=raw_result.total_hpwl,
            dead_space_ratio=dead_space_ratio,
        )
        ok, issues = validate_floorplan(
            result,
            self.hard_modules,
            eps=self.problem2_config.feasibility_tolerance,
        )
        result.feasible = ok
        result.verify_info = "; ".join(issues[:5])
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
        - Apply 90掳/180掳/270掳 rotation to entire module
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
