"""Runner for Problem 1 artifacts."""

import os

from ..problems import problem1
from .artifacts import write_result_summary


def run(solver, chip_name: str, output_root: str):
    """Solve Problem 1 and write its artifacts under output_root/problem1."""
    problem_dir = os.path.join(output_root, "problem1")
    result = problem1.solve(solver, save_dir=problem_dir, chip_name=chip_name)
    summary_path = os.path.join(problem_dir, f"{chip_name}_problem1_summary.txt")
    write_result_summary(result, summary_path)
    return result
