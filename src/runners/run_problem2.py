"""Runner for Problem 2 artifacts."""

import os

from ..core.visualize import visualize_floorplan
from ..problems import problem2
from .artifacts import write_result_summary


def run(solver, chip_name: str, output_root: str, dead_space_ratio: float = 0.15):
    """Solve Problem 2 and write its artifacts under output_root/problem2."""
    problem_dir = os.path.join(output_root, "problem2")
    os.makedirs(problem_dir, exist_ok=True)

    result = problem2.solve(solver, dead_space_ratio=dead_space_ratio)

    summary_path = os.path.join(problem_dir, f"{chip_name}_problem2_summary.txt")
    image_path = os.path.join(problem_dir, f"{chip_name}_problem2_floorplan.png")
    write_result_summary(result, summary_path)
    visualize_floorplan(
        result,
        solver.hard_modules,
        title=f"{chip_name} - Problem 2: HPWL Minimization",
        save_path=image_path,
        show_nets=True,
        nets=solver.nets,
        terminal_positions=solver.terminal_positions,
    )
    return result
