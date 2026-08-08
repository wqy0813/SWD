"""Problem 1 entry points: variable-outline area minimization."""

from typing import Optional

from ..core.models import FloorplanResult
from .solver import ProblemSolver


def solve(solver: ProblemSolver, save_dir: Optional[str] = None,
          chip_name: str = "chip") -> FloorplanResult:
    """Solve Problem 1 without exposing other problem flows."""
    return solver.solve_problem1(save_dir=save_dir, chip_name=chip_name)
