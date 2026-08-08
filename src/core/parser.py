# 本文件负责读取题目附件中的 .blocks、.nets 和 .pl 文件，
# 将原始文本数据解析为模块、连线网络和固定端点坐标。
"""VLSI floorplanning solver modules split from the original spr backup."""


import re
from typing import Dict, List, Tuple

from .models import Module, Net, SubBlock

class VLSIParser:
    """Parses .blocks, .nets, and .pl files."""

    @staticmethod
    def parse_blocks(filepath: str) -> List[Module]:
        """Parse .blocks file. Returns list of Module objects."""
        modules = []
        # Try multiple encodings
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Parse lines
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('Num'):
                # Skip header lines
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            name = parts[0]
            mtype = parts[1]

            if mtype == 'terminal':
                mod = Module(name=name, module_type='terminal',
                             width=0, height=0, is_hard=False)
                modules.append(mod)

            elif mtype == 'block':
                # Format: b0 block 4 (0,0) (0,82) (199,82) (199,0)
                coords = []
                if len(parts) >= 3:
                    try:
                        num_vertices = int(parts[2])
                    except ValueError:
                        continue
                    coord_matches = re.findall(
                        r'\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)',
                        line
                    )
                    for x_str, y_str in coord_matches[:num_vertices]:
                        coords.append((float(x_str), float(y_str)))

                if coords:
                    # Determine shape type and compute bounding box
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)

                    mod = Module(name=name, module_type='block',
                                 width=width, height=height, is_hard=True)

                    # Detect non-rectangular shapes (Problem 4)
                    if len(coords) >= 6:
                        # Check if it's L-shaped or T-shaped
                        shape_type = VLSIParser._detect_shape_type(coords)
                        mod.shape_type = shape_type
                        if shape_type != 'rect':
                            mod.sub_blocks = VLSIParser._decompose_shape(coords)

                    modules.append(mod)

            elif mtype == 'terminal':
                mod = Module(name=name, module_type='terminal',
                             width=0, height=0, is_hard=False)
                modules.append(mod)

        return modules

    @staticmethod
    def _detect_shape_type(coords: List[Tuple[float, float]]) -> str:
        """Detect if a polygon is L-shaped, T-shaped, or rectangular."""
        # Normalize coordinates relative to min x, min y
        min_x = min(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_x = max(c[0] for c in coords)
        max_y = max(c[1] for c in coords)

        w = max_x - min_x
        h = max_y - min_y

        # Check if the polygon covers the full bounding box (rectangular)
        # If area of polygon < bounding box area, it's non-rectangular
        area_poly = VLSIParser._polygon_area(coords)
        area_bbox = w * h

        if abs(area_poly - area_bbox) < 1e-6:
            return 'rect'

        # Determine specific shape by checking which corners are missing
        # Simplified: check if one quadrant is missing (L-shape)
        # or if it's more complex (T-shape)
        mid_x = min_x + w / 2
        mid_y = min_y + h / 2

        quadrants_covered = 0
        quadrants = [
            [(min_x, min_y), (mid_x, min_y), (mid_x, mid_y), (min_x, mid_y)],
            [(mid_x, min_y), (max_x, min_y), (max_x, mid_y), (mid_x, mid_y)],
            [(min_x, mid_y), (mid_x, mid_y), (mid_x, max_y), (min_x, max_y)],
            [(mid_x, mid_y), (max_x, mid_y), (max_x, max_y), (mid_x, max_y)],
        ]

        for quad in quadrants:
            if VLSIParser._polygon_overlaps_rect(coords, quad):
                quadrants_covered += 1

        if quadrants_covered == 3:
            return 'L'
        else:
            return 'T'

    @staticmethod
    def _polygon_area(coords: List[Tuple[float, float]]) -> float:
        """Shoelace formula for polygon area."""
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _polygon_overlaps_rect(coords, rect) -> bool:
        """Check if polygon overlaps with a rectangle (simplified)."""
        # Check if any vertex of polygon is inside rect
        rx = [p[0] for p in rect]
        ry = [p[1] for p in rect]
        rminx, rmaxx = min(rx), max(rx)
        rminy, rmaxy = min(ry), max(ry)

        rect_center = ((rminx + rmaxx) / 2, (rminy + rmaxy) / 2)
        # Check if polygon overlaps rect center
        # Simplified ray casting
        return VLSIParser._point_in_polygon(rect_center[0], rect_center[1], coords)

    @staticmethod
    def _point_in_polygon(x: float, y: float, coords) -> bool:
        """Ray casting algorithm."""
        n = len(coords)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = coords[i][0], coords[i][1]
            xj, yj = coords[j][0], coords[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _decompose_shape(coords: List[Tuple[float, float]]) -> List[SubBlock]:
        """Decompose non-rectangular shape into sub-rectangles."""
        # Normalize
        min_x = min(c[0] for c in coords)
        min_y = min(c[1] for c in coords)

        # Use a simple grid-based decomposition
        # For L-shaped: decompose into 2 rectangles
        # For T-shaped: decompose into 3 rectangles

        xs = sorted(set(c[0] for c in coords))
        ys = sorted(set(c[1] for c in coords))

        sub_blocks = []
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                x1, x2 = xs[i], xs[i + 1]
                y1, y2 = ys[j], ys[j + 1]
                # Check if this cell is inside the polygon
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if VLSIParser._point_in_polygon(center_x, center_y, coords):
                    sub_blocks.append(SubBlock(
                        rel_x=x1 - min_x,
                        rel_y=y1 - min_y,
                        width=x2 - x1,
                        height=y2 - y1
                    ))

        # Merge adjacent sub-blocks for efficiency
        return VLSIParser._merge_sub_blocks(sub_blocks)

    @staticmethod
    def _merge_sub_blocks(blocks: List[SubBlock]) -> List[SubBlock]:
        """Merge adjacent sub-blocks if they share a full edge."""
        if len(blocks) <= 1:
            return blocks
        # Simple greedy merge
        merged = True
        result = list(blocks)
        while merged:
            merged = False
            new_result = []
            used = set()
            for i, a in enumerate(result):
                if i in used:
                    continue
                found_merge = False
                for j, b in enumerate(result):
                    if j <= i or j in used:
                        continue
                    # Check if same height and adjacent horizontally
                    if (abs(a.height - b.height) < 1e-6 and
                        abs((a.rel_x + a.width) - b.rel_x) < 1e-6 and
                        abs(a.rel_y - b.rel_y) < 1e-6):
                        new_result.append(SubBlock(a.rel_x, a.rel_y,
                                                   a.width + b.width, a.height))
                        used.add(i)
                        used.add(j)
                        found_merge = True
                        merged = True
                        break
                    # Check if same width and adjacent vertically
                    if (abs(a.width - b.width) < 1e-6 and
                        abs((a.rel_y + a.height) - b.rel_y) < 1e-6 and
                        abs(a.rel_x - b.rel_x) < 1e-6):
                        new_result.append(SubBlock(a.rel_x, a.rel_y,
                                                   a.width, a.height + b.height))
                        used.add(i)
                        used.add(j)
                        found_merge = True
                        merged = True
                        break
                if not found_merge:
                    new_result.append(a)
            result = new_result
            if not merged:
                break
        return result

    @staticmethod
    def parse_nets(filepath: str) -> List[Net]:
        """Parse .nets file."""
        nets = []
        modules_set = set()
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            return nets  # Could not read file

        num_nets = 0
        num_pins = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith('//'):
                continue
            if 'NumNets' in line or 'NumNets' in line:
                continue
            if 'NumPins' in line or 'NumPins' in line:
                continue
            if 'NetDegree' in line or 'NetDegree' in line:
                # Parse degree number
                try:
                    if ':' in line or '：' in line:
                        sep = ':' if ':' in line else '：'
                        degree = int(line.split(sep)[1].strip())
                    else:
                        degree = int(line.split()[-1])
                except (ValueError, IndexError):
                    continue
                net = Net()
                for _ in range(degree):
                    if i < len(lines):
                        pin_name = lines[i].strip()
                        i += 1
                        if pin_name and not pin_name.startswith('//'):
                            net.pins.append(pin_name)
                if net.pins:
                    nets.append(net)

        return nets

    @staticmethod
    def parse_pl(filepath: str) -> Dict[str, Tuple[float, float]]:
        """Parse .pl file (terminal positions)."""
        positions = {}
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            return positions
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    positions[name] = (x, y)
                except ValueError:
                    continue
        return positions


# ============================================================
# B*-TREE REPRESENTATION
# ============================================================
