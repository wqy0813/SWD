"""Problem 2 entry points: fixed-square-outline HPWL minimization."""

import os
from typing import Optional

from ..core.models import FloorplanResult
from .solver import ProblemSolver


def solve(solver: ProblemSolver,
          dead_space_ratio: Optional[float] = None) -> FloorplanResult:
    """Solve Problem 2 without exposing other problem flows."""
    return solver.solve_problem2(dead_space_ratio=dead_space_ratio)


def solve_problem2_chip(problem_path: str, chip_name: str,
                        output_root: str = "outputs",
                        dead_space_ratio: float = 0.15) -> FloorplanResult:
    """Solve Problem 2 for one chip dataset and write standard artifacts."""
    blocks_file = os.path.join(problem_path, f"{chip_name}.blocks")
    nets_file = os.path.join(problem_path, f"{chip_name}.nets")
    pl_file = os.path.join(problem_path, f"{chip_name}.pl")

    solver = ProblemSolver(blocks_file, nets_file, pl_file)

    # Imported lazily to keep problem modules usable without runner side effects.
    from ..runners.run_problem2 import run

    return run(
        solver,
        chip_name,
        os.path.join(output_root, chip_name),
        dead_space_ratio=dead_space_ratio,
    )
