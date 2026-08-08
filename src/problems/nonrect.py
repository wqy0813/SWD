# 本文件实现问题 4 的 L 型、T 型等异形模块处理逻辑，
# 包括旋转变体生成、非矩形模块碰撞检测和小规模异形布局搜索。
"""VLSI floorplanning solver modules split from the original spr backup."""


import copy
import math
import random
from typing import List, Tuple

from ..core.models import FloorplanResult, Module, SubBlock

class NonRectSolver:
    """
    Extended solver for Problem 4: L-shaped and T-shaped modules.

    Uses a direct approach for small problem instances (like the 4-module example)
    with exhaustive or grid-based search.
    """

    @staticmethod
    def solve_4_modules(modules: List[Module]) -> FloorplanResult:
        """
        Solve the 4-module problem (1 T-shape, 1 L-shape, 2 rectangles) from Problem 4.

        Approach:
        - Represent each module as a bitmap/polygon
        - Use grid-based placement with Simulated Annealing
        - Each module can be rotated 90°, 180°, 270°
        """
        print("\n" + "=" * 60)
        print("PROBLEM 4: Solving 4-module instance")
        print("=" * 60)

        # Generate all rotation variants for each module
        variants = []
        for mod in modules:
            mod_variants = NonRectSolver._generate_rotations(mod)
            variants.append(mod_variants)
            print(f"  {mod.name}: {mod.shape_type}-shaped, "
                  f"{len(mod_variants)} rotation variants")

        # Simulated Annealing on sequence-pair-like representation
        best_result = NonRectSolver._sa_nonrect(modules, variants)

        print(f"Best area: {best_result.area:.2f}")
        print(f"Outline: {best_result.outline_width:.2f} x {best_result.outline_height:.2f}")

        return best_result

    @staticmethod
    def _generate_rotations(mod: Module) -> List[List[SubBlock]]:
        """Generate all rotation variants of a module (0°, 90°, 180°, 270°)."""
        variants = []

        if mod.shape_type == 'rect':
            # Rectangular: swap or not
            variants.append([SubBlock(0, 0, mod.width, mod.height)])
            if abs(mod.width - mod.height) > 1e-9:
                variants.append([SubBlock(0, 0, mod.height, mod.width)])
        else:
            # Non-rect: rotate each sub-block
            for angle in [0, 90, 180, 270]:
                rotated = NonRectSolver._rotate_sub_blocks(mod.sub_blocks, angle)
                if rotated not in variants:
                    variants.append(rotated)
            # Deduplicate
            unique_variants = []
            for v in variants:
                is_dup = False
                for uv in unique_variants:
                    if NonRectSolver._same_config(v, uv):
                        is_dup = True
                        break
                if not is_dup:
                    unique_variants.append(v)
            variants = unique_variants

        return variants

    @staticmethod
    def _rotate_sub_blocks(blocks: List[SubBlock], angle: int) -> List[SubBlock]:
        """Rotate sub-blocks by angle (counter-clockwise)."""
        if angle == 0:
            return [SubBlock(b.rel_x, b.rel_y, b.width, b.height) for b in blocks]

        # Compute bounding box
        max_x = max(b.rel_x + b.width for b in blocks)
        max_y = max(b.rel_y + b.height for b in blocks)

        result = []
        for b in blocks:
            cx = b.rel_x + b.width / 2.0
            cy = b.rel_y + b.height / 2.0

            if angle == 90:
                # (x, y) -> (max_y - y - h, x)
                nx = max_y - b.rel_y - b.height
                ny = b.rel_x
                nw = b.height
                nh = b.width
            elif angle == 180:
                # (x, y) -> (max_x - x - w, max_y - y - h)
                nx = max_x - b.rel_x - b.width
                ny = max_y - b.rel_y - b.height
                nw = b.width
                nh = b.height
            elif angle == 270:
                # (x, y) -> (y, max_x - x - w)
                nx = b.rel_y
                ny = max_x - b.rel_x - b.width
                nw = b.height
                nh = b.width

            result.append(SubBlock(nx, ny, nw, nh))

        # Normalize to origin
        min_x = min(b.rel_x for b in result)
        min_y = min(b.rel_y for b in result)
        result = [SubBlock(b.rel_x - min_x, b.rel_y - min_y, b.width, b.height)
                  for b in result]

        return result

    @staticmethod
    def _same_config(a: List[SubBlock], b: List[SubBlock]) -> bool:
        """Check if two sub-block configurations are the same."""
        if len(a) != len(b):
            return False
        # Normalize and compare
        a_norm = sorted([(sb.rel_x, sb.rel_y, sb.width, sb.height) for sb in a])
        b_norm = sorted([(sb.rel_x, sb.rel_y, sb.width, sb.height) for sb in b])
        return all(abs(a_norm[i][j] - b_norm[i][j]) < 1e-9
                   for i in range(len(a_norm)) for j in range(4))

    @staticmethod
    def _sa_nonrect(modules: List[Module],
                    variants: List[List[SubBlock]]) -> FloorplanResult:
        """Simulated Annealing for non-rectangular modules."""
        n = len(modules)
        # Each module: position (x, y) + variant index
        current_variants = [0] * n
        current_x = [0.0] * n
        current_y = [0.0] * n

        # Initialize in a spread-out grid to avoid all starting at origin
        grid_cols = int(math.ceil(math.sqrt(n)))
        for i in range(n):
            current_variants[i] = random.randrange(len(variants[i]))
            # Spread modules in a grid pattern
            col = i % grid_cols
            row = i // grid_cols
            current_x[i] = col * 50.0 + random.uniform(0, 10)
            current_y[i] = row * 50.0 + random.uniform(0, 10)

        # Compact to origin (push bottom-left)
        current_x, current_y = NonRectSolver._compact_placement(
            modules, variants, current_variants, current_x, current_y)

        current_width, current_height = NonRectSolver._compute_bbox(
            modules, variants, current_variants, current_x, current_y)
        current_area = current_width * current_height

        def objective(w, h):
            ar = max(w, h) / max(1e-9, min(w, h))
            return w * h * (1.0 + 0.1 * abs(ar - 1.0))

        current_cost = objective(current_width, current_height)

        best_variants = current_variants[:]
        best_x = current_x[:]
        best_y = current_y[:]
        best_cost = current_cost
        best_width = current_width
        best_height = current_height

        T = 10000.0
        T_final = 0.01
        cooling_rate = 0.96
        max_iter = 8000

        for iteration in range(max_iter):
            if T < T_final:
                break

            for _ in range(30):
                i = random.randrange(n)
                j = random.randrange(n)  # secondary index for swaps

                # Save state for BOTH i and j (j might be affected by swap)
                old_var_i = current_variants[i]
                old_x_i = current_x[i]
                old_y_i = current_y[i]
                old_var_j = current_variants[j]
                old_x_j = current_x[j]
                old_y_j = current_y[j]

                # Perturb
                perturb_type = random.random()
                is_swap = False
                if perturb_type < 0.25:
                    # Change variant (rotation)
                    if len(variants[i]) > 1:
                        new_var = (current_variants[i] + random.randrange(1, len(variants[i]))) % len(variants[i])
                        current_variants[i] = new_var
                elif perturb_type < 0.5:
                    # Swap positions with another module
                    if i != j:
                        current_x[i], current_x[j] = current_x[j], current_x[i]
                        current_y[i], current_y[j] = current_y[j], current_y[i]
                        # Swap variants, but clamp to valid range for each module
                        vi, vj = current_variants[j], current_variants[i]
                        current_variants[i] = min(vi, len(variants[i]) - 1)
                        current_variants[j] = min(vj, len(variants[j]) - 1)
                        is_swap = True
                elif perturb_type < 0.75:
                    # Move in x
                    step = max(1.0, current_width * 0.2)
                    current_x[i] += random.uniform(-step, step)
                    if current_x[i] < 0:
                        current_x[i] = 0
                else:
                    # Move in y
                    step = max(1.0, current_height * 0.2)
                    current_y[i] += random.uniform(-step, step)
                    if current_y[i] < 0:
                        current_y[i] = 0

                # Compact (push to bottom-left)
                cx, cy = NonRectSolver._compact_placement(
                    modules, variants, current_variants, current_x, current_y)

                # Recompute
                nw, nh = NonRectSolver._compute_bbox(
                    modules, variants, current_variants, cx, cy)
                nc = objective(nw, nh)

                delta = nc - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_x = cx
                    current_y = cy
                    current_cost = nc
                    current_width = nw
                    current_height = nh

                    if current_cost < best_cost:
                        best_variants = current_variants[:]
                        best_x = current_x[:]
                        best_y = current_y[:]
                        best_cost = current_cost
                        best_width = nw
                        best_height = nh
                else:
                    # Reject - restore both i and j
                    current_variants[i] = old_var_i
                    current_x[i] = old_x_i
                    current_y[i] = old_y_i
                    if is_swap:
                        current_variants[j] = old_var_j
                        current_x[j] = old_x_j
                        current_y[j] = old_y_j

            T *= cooling_rate

        # Build result with variant index stored as rotation info
        positions = {}
        for i, mod in enumerate(modules):
            positions[mod.name] = (best_x[i], best_y[i],
                                   best_variants[i])  # Store actual variant index

        return FloorplanResult(
            module_positions=positions,
            outline_width=best_width,
            outline_height=best_height,
            area=best_width * best_height,
            aspect_ratio=max(best_width, best_height) /
                         max(1e-9, min(best_width, best_height))
        )

    @staticmethod
    def _compact_placement(modules: List[Module],
                           variants: List[List[SubBlock]],
                           var_indices: List[int],
                           xs: List[float],
                           ys: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compact placement: resolve all overlaps and push to origin.
        Uses a simple left-to-right, bottom-to-top packing similar to
        1D bin packing after sorting by initial position.
        """
        n = len(modules)
        # Order: sort by initial (x, y)
        order = sorted(range(n), key=lambda i: (xs[i], ys[i]))

        new_xs = [0.0] * n
        new_ys = [0.0] * n

        for rank, idx in enumerate(order):
            blocks = variants[idx][var_indices[idx]]

            if rank == 0:
                new_xs[idx] = 0.0
                new_ys[idx] = 0.0
                continue

            # Try grid positions based on placed modules' corners
            candidates = [(0.0, 0.0)]

            for j in range(n):
                if j == idx:
                    continue
                if rank == 0 and j != order[0]:
                    continue
                if j not in order[:rank]:
                    continue  # not placed yet

                j_blocks = variants[j][var_indices[j]]
                j_right = new_xs[j] + max(b.rel_x + b.width for b in j_blocks)
                j_top = new_ys[j] + max(b.rel_y + b.height for b in j_blocks)

                # Corners of the bounding box of j
                candidates.append((j_right, new_ys[j]))
                candidates.append((new_xs[j], j_top))

                # Sub-block corners
                for sb in j_blocks:
                    candidates.append((new_xs[j] + sb.rel_x + sb.width, new_ys[j] + sb.rel_y))
                    candidates.append((new_xs[j] + sb.rel_x, new_ys[j] + sb.rel_y + sb.height))

            # Sort by x+y (bottom-left preference)
            candidates.sort(key=lambda p: p[0] + p[1])

            best_pos = None
            for cx, cy in candidates:
                cx = max(0.0, cx)
                cy = max(0.0, cy)

                # Slide left from cx
                slide_x = cx
                step = 5.0
                while slide_x > 0:
                    test_x = max(0.0, slide_x - step)
                    collision = False
                    for j in order[:rank]:
                        if NonRectSolver._modules_overlap(
                                test_x, cy, blocks,
                                new_xs[j], new_ys[j], variants[j][var_indices[j]]):
                            collision = True
                            break
                    if collision:
                        break
                    slide_x = test_x
                # Fine-tune
                while slide_x > 0:
                    test_x = slide_x - 1.0
                    collision = False
                    for j in order[:rank]:
                        if NonRectSolver._modules_overlap(
                                test_x, cy, blocks,
                                new_xs[j], new_ys[j], variants[j][var_indices[j]]):
                            collision = True
                            break
                    if collision:
                        break
                    slide_x = test_x

                # Slide down from cy
                slide_y = cy
                step = 5.0
                while slide_y > 0:
                    test_y = max(0.0, slide_y - step)
                    collision = False
                    for j in order[:rank]:
                        if NonRectSolver._modules_overlap(
                                slide_x, test_y, blocks,
                                new_xs[j], new_ys[j], variants[j][var_indices[j]]):
                            collision = True
                            break
                    if collision:
                        break
                    slide_y = test_y
                # Fine-tune
                while slide_y > 0:
                    test_y = slide_y - 1.0
                    collision = False
                    for j in order[:rank]:
                        if NonRectSolver._modules_overlap(
                                slide_x, test_y, blocks,
                                new_xs[j], new_ys[j], variants[j][var_indices[j]]):
                            collision = True
                            break
                    if collision:
                        break
                    slide_y = test_y

                # Verify final position
                collision = False
                for j in order[:rank]:
                    if NonRectSolver._modules_overlap(
                            slide_x, slide_y, blocks,
                            new_xs[j], new_ys[j], variants[j][var_indices[j]]):
                        collision = True
                        break

                if not collision:
                    if best_pos is None or (slide_x + slide_y) < (best_pos[0] + best_pos[1]):
                        best_pos = (slide_x, slide_y)

            if best_pos is None:
                # Fallback: place at end of x-axis
                max_right = 0.0
                for j in order[:rank]:
                    j_blocks = variants[j][var_indices[j]]
                    j_right = new_xs[j] + max(b.rel_x + b.width for b in j_blocks)
                    max_right = max(max_right, j_right)
                best_pos = (max_right + 1.0, 0.0)

            new_xs[idx] = best_pos[0]
            new_ys[idx] = best_pos[1]

        return new_xs, new_ys

    @staticmethod
    def _modules_overlap(x1: float, y1: float, blocks1: List[SubBlock],
                         x2: float, y2: float, blocks2: List[SubBlock]) -> bool:
        """Check if two modules overlap."""
        for b1 in blocks1:
            bx1 = x1 + b1.rel_x
            by1 = y1 + b1.rel_y
            bx1_e = bx1 + b1.width
            by1_e = by1 + b1.height
            for b2 in blocks2:
                bx2 = x2 + b2.rel_x
                by2 = y2 + b2.rel_y
                bx2_e = bx2 + b2.width
                by2_e = by2 + b2.height
                if (bx1 < bx2_e and bx1_e > bx2 and
                        by1 < by2_e and by1_e > by2):
                    return True
        return False

    @staticmethod
    def _compute_bbox(modules: List[Module],
                      variants: List[List[SubBlock]],
                      var_indices: List[int],
                      xs: List[float],
                      ys: List[float]) -> Tuple[float, float]:
        """Compute bounding box of current placement."""
        max_x = 0.0
        max_y = 0.0
        for i, mod in enumerate(modules):
            sub_blocks = variants[i][var_indices[i]]
            for sb in sub_blocks:
                max_x = max(max_x, xs[i] + sb.rel_x + sb.width)
                max_y = max(max_y, ys[i] + sb.rel_y + sb.height)
        return max_x, max_y


# ============================================================
# VISUALIZATION
# ============================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon, Rectangle
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not installed — visualization disabled.")
    print("         Install with: pip install matplotlib")
