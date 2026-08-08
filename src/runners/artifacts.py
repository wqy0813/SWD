"""Shared output helpers for experiment runners."""

import os


def write_result_summary(result, save_path: str):
    """Write a compact text summary beside generated floorplan figures."""
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"outline_width: {result.outline_width:.6f}\n")
        f.write(f"outline_height: {result.outline_height:.6f}\n")
        f.write(f"area: {result.area:.6f}\n")
        f.write(f"aspect_ratio: {result.aspect_ratio:.6f}\n")
        f.write(f"dead_space_ratio: {result.dead_space_ratio:.6f}\n")
        f.write(f"total_hpwl: {result.total_hpwl:.6f}\n")
        f.write(f"feasible: {result.feasible}\n")
        if result.verify_info:
            f.write(f"verify_info: {result.verify_info}\n")
        f.write("\nmodule,x,y,rotated\n")
        for name, (x, y, rotated) in sorted(result.module_positions.items()):
            f.write(f"{name},{x:.6f},{y:.6f},{rotated}\n")
