# 本文件提供命令行入口，支持运行样例或指定 data 目录中的芯片实例，
# 并串联问题求解与可视化输出，便于从终端一键执行实验。
"""Command line entry points for the VLSI floorplanning solver."""


import argparse
import os

from .models import Module, SubBlock
from .nonrect import NonRectSolver
from .solver import ProblemSolver
from .visualize import visualize_floorplan

def solve_chip(problem_path: str, chip_name: str):
    """
    Solve all problems for a single chip dataset.

    Args:
        problem_path: directory containing .blocks, .nets, .pl files
        chip_name: name of the chip (e.g., "n100")
    """
    blocks_file = os.path.join(problem_path, f"{chip_name}.blocks")
    nets_file = os.path.join(problem_path, f"{chip_name}.nets")
    pl_file = os.path.join(problem_path, f"{chip_name}.pl")

    # Check alternate naming
    if not os.path.exists(blocks_file):
        blocks_file = os.path.join(problem_path, f"{chip_name}.blocks")
    if not os.path.exists(blocks_file):
        # Try listing directory
        for f in os.listdir(problem_path):
            if f.endswith('.blocks'):
                blocks_file = os.path.join(problem_path, f)
                base = f[:-7]
                nets_file = os.path.join(problem_path, f"{base}.nets")
                pl_file = os.path.join(problem_path, f"{base}.pl")
                break

    print(f"\n{'#' * 70}")
    print(f"# Solving for {chip_name}")
    print(f"# Blocks: {blocks_file}")
    print(f"# Nets: {nets_file}")
    print(f"# PL: {pl_file}")
    print(f"{'#' * 70}")

    solver = ProblemSolver(blocks_file, nets_file, pl_file)

    # Problem 1
    result1 = solver.solve_problem1()
    visualize_floorplan(result1, solver.hard_modules,
                        title=f"{chip_name} - Problem 1: Area Minimization",
                        save_path=f"{chip_name}_problem1.png")

    # Problem 2
    result2 = solver.solve_problem2(dead_space_ratio=0.15)
    visualize_floorplan(result2, solver.hard_modules,
                        title=f"{chip_name} - Problem 2: HPWL Minimization",
                        save_path=f"{chip_name}_problem2.png",
                        show_nets=True,
                        nets=solver.nets,
                        terminal_positions=solver.terminal_positions)

    # Problem 3
    min_dsr, result3 = solver.solve_problem3()
    visualize_floorplan(result3, solver.hard_modules,
                        title=f"{chip_name} - Problem 3: Min DSR = {min_dsr:.4f}",
                        save_path=f"{chip_name}_problem3.png",
                        show_nets=True,
                        nets=solver.nets,
                        terminal_positions=solver.terminal_positions)

    # Problem 4 (only for specially defined non-rect modules)
    # Will be solved separately

    return {
        'chip': chip_name,
        'problem1': result1,
        'problem2': result2,
        'problem3': result3,
        'min_dsr': min_dsr
    }



def solve_problem4_example():
    """
    Solve Problem 4 with the example from Figure 3.

    Expected modules: 1 T-shaped, 1 L-shaped, 2 rectangular
    Dimensions from the problem figure (approximate, users should adjust).
    """
    print("\n" + "#" * 70)
    print("# PROBLEM 4: Non-Rectangular Module Example")
    print("#" * 70)

    # Create the example modules from Figure 3
    # Users should adjust these coordinates based on the actual figure
    # Example T-shape module
    t_shape = Module(name="T1", module_type='block', width=60, height=40,
                     is_hard=True, shape_type='T')
    t_shape.sub_blocks = [
        SubBlock(0, 0, 60, 10),    # top bar
        SubBlock(25, 10, 10, 30),  # stem
    ]

    # Example L-shape module
    l_shape = Module(name="L1", module_type='block', width=40, height=50,
                     is_hard=True, shape_type='L')
    l_shape.sub_blocks = [
        SubBlock(0, 0, 40, 15),    # wide part
        SubBlock(0, 15, 15, 35),   # tall part
    ]

    # Example rectangle modules
    rect1 = Module(name="R1", module_type='block', width=30, height=25,
                   is_hard=True, shape_type='rect')
    rect2 = Module(name="R2", module_type='block', width=35, height=20,
                   is_hard=True, shape_type='rect')

    modules = [t_shape, l_shape, rect1, rect2]

    result = NonRectSolver.solve_4_modules(modules)

    # Generate rotation variants for visualization
    variants = [NonRectSolver._generate_rotations(m) for m in modules]

    # Visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[SKIP] Cannot save problem4_example.png - matplotlib not installed")
        return result

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    patterns = ['///', '\\\\\\', '|||', '---']

    for i, mod in enumerate(modules):
        if mod.name in result.module_positions:
            x, y, var_idx = result.module_positions[mod.name]
            var_idx = int(var_idx)  # variant index (not just boolean)
            color = colors[i % len(colors)]

            # Use the correct variant's sub-blocks
            active_blocks = variants[i][var_idx]

            bb_min_x = float('inf')
            bb_min_y = float('inf')
            bb_max_x = float('-inf')
            bb_max_y = float('-inf')

            for sb in active_blocks:
                sbx = x + sb.rel_x
                sby = y + sb.rel_y
                rect = Rectangle((sbx, sby),
                                 sb.width, sb.height, linewidth=1.5,
                                 edgecolor='black', facecolor=color, alpha=0.7,
                                 hatch=patterns[i % len(patterns)])
                ax.add_patch(rect)
                bb_min_x = min(bb_min_x, sbx)
                bb_min_y = min(bb_min_y, sby)
                bb_max_x = max(bb_max_x, sbx + sb.width)
                bb_max_y = max(bb_max_y, sby + sb.height)

            cx = (bb_min_x + bb_max_x) / 2.0
            cy = (bb_min_y + bb_max_y) / 2.0
            ax.text(cx, cy, f"{mod.name}\n({mod.shape_type})",
                    ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw outline
    outline = Rectangle((0, 0), result.outline_width, result.outline_height,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--')
    ax.add_patch(outline)

    ax.set_xlim(-5, result.outline_width * 1.2)
    ax.set_ylim(-5, result.outline_height * 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Problem 4: Non-Rectangular Module Floorplan\n"
                 f"Area = {result.area:.2f}")

    plt.tight_layout()
    plt.savefig("problem4_example.png", dpi=150, bbox_inches='tight')
    print(f"Saved visualization to problem4_example.png")
    plt.close()

    return result


# ============================================================
# STANDALONE TEST / DEMO
# ============================================================


def create_sample_data():
    """Create sample data files for testing when real data is not available."""
    import os

    # Create sample blocks, nets, and pl files for testing
    sample_dir = "sample_data"
    os.makedirs(sample_dir, exist_ok=True)

    # Create a small test case: 10 hard blocks + 5 terminals
    blocks_content = """NumHardBlocks 10
NumTerminals 5
b0 block 4 (0,0) (0,20) (30,20) (30,0)
b1 block 4 (0,0) (0,15) (25,15) (25,0)
b2 block 4 (0,0) (0,25) (20,25) (20,0)
b3 block 4 (0,0) (0,18) (35,18) (35,0)
b4 block 4 (0,0) (0,22) (28,22) (28,0)
b5 block 4 (0,0) (0,12) (32,12) (32,0)
b6 block 4 (0,0) (0,30) (22,30) (22,0)
b7 block 4 (0,0) (0,16) (26,16) (26,0)
b8 block 4 (0,0) (0,24) (18,24) (18,0)
b9 block 4 (0,0) (0,28) (24,28) (24,0)
p1 terminal
p2 terminal
p3 terminal
p4 terminal
p5 terminal
"""
    with open(os.path.join(sample_dir, "test.blocks"), 'w', encoding='utf-8') as f:
        f.write(blocks_content)

    nets_content = """NumNets 8
NumPins 30
NetDegree 3
p1
b0
b1
NetDegree 4
p2
b2
b3
b4
NetDegree 3
p3
b5
b6
NetDegree 4
p4
b7
b8
b9
NetDegree 3
p1
b2
b5
NetDegree 3
p5
b0
b3
NetDegree 4
p2
b1
b6
b8
NetDegree 5
p3
b0
b4
b7
b9
"""
    with open(os.path.join(sample_dir, "test.nets"), 'w', encoding='utf-8') as f:
        f.write(nets_content)

    pl_content = """p1 100 200
p2 300 150
p3 50 350
p4 400 100
p5 250 300
"""
    with open(os.path.join(sample_dir, "test.pl"), 'w', encoding='utf-8') as f:
        f.write(pl_content)

    print(f"Created sample data in {sample_dir}/")
    return sample_dir




def main():
    parser = argparse.ArgumentParser(
        description="VLSI Floorplanning Solver for Problems 1-4")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to directory containing .blocks, .nets, .pl files")
    parser.add_argument("--chip", type=str, default=None,
                        help="Chip name (e.g., n100, n200, n300)")
    parser.add_argument("--problem", type=int, choices=[0, 1, 2, 3, 4], default=0,
                        help="Problem number to solve (0 = solve all)")
    parser.add_argument("--sample", action="store_true",
                        help="Create sample data and run demo")

    args = parser.parse_args()

    if args.sample:
        sample_dir = create_sample_data()
        solver = ProblemSolver(
            os.path.join(sample_dir, "test.blocks"),
            os.path.join(sample_dir, "test.nets"),
            os.path.join(sample_dir, "test.pl")
        )

        print("\n\n" + "=" * 60)
        print("RUNNING DEMO WITH SAMPLE DATA")
        print("=" * 60)

        if args.problem in (0, 1):
            result1 = solver.solve_problem1()
            visualize_floorplan(result1, solver.hard_modules,
                                title="Sample - Problem 1: Area Minimization",
                                save_path="sample_problem1.png")

        if args.problem in (0, 2):
            result2 = solver.solve_problem2(dead_space_ratio=0.15)
            visualize_floorplan(result2, solver.hard_modules,
                                title="Sample - Problem 2: HPWL Minimization",
                                save_path="sample_problem2.png",
                                show_nets=True,
                                nets=solver.nets,
                                terminal_positions=solver.terminal_positions)

        if args.problem in (0, 3):
            min_dsr, result3 = solver.solve_problem3()
            visualize_floorplan(result3, solver.hard_modules,
                                title=f"Sample - Problem 3: Min DSR = {min_dsr:.4f}",
                                save_path="sample_problem3.png",
                                show_nets=True,
                                nets=solver.nets,
                                terminal_positions=solver.terminal_positions)

        if args.problem in (0, 4):
            solve_problem4_example()

    elif args.path and args.chip:
        results = solve_chip(args.path, args.chip)

        print("\n\n" + "=" * 60)
        print("SUMMARY OF RESULTS")
        print("=" * 60)
        print(f"\nChip: {results['chip']}")
        print(f"  Problem 1 - Area: {results['problem1'].area:.2f}, "
              f"AR: {results['problem1'].aspect_ratio:.4f}")
        print(f"  Problem 2 - HPWL: {results['problem2'].total_hpwl:.2f}")
        print(f"  Problem 3 - Min DSR: {results['min_dsr']:.4f}, "
              f"HPWL: {results['problem3'].total_hpwl:.2f}")

    else:
        print("VLSI Floorplanning Solver")
        print("=" * 60)
        print("\nUsage:")
        print("  python -m src.main --path <data_dir> --chip <n100|n200|n300>")
        print("  python -m src.main --sample  # Run with generated sample data")


if __name__ == "__main__":
    main()
