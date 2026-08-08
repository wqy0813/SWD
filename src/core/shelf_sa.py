"""Shelf-packing simulated annealing used for fixed-outline feasibility."""

import math
import random
from typing import Dict, List, Optional, Tuple

from .models import Module, Net


class ShelfSA:
    """Simulated annealing over module order and rotations with shelf decoding."""

    def __init__(self, modules: List[Module], nets: Optional[List[Net]] = None,
                 terminal_positions: Optional[Dict[str, Tuple[float, float]]] = None,
                 seed: int = 42):
        self.modules = modules
        self.n = len(modules)
        self.nets = nets or []
        self.terminal_positions = terminal_positions or {}
        self.rng = random.Random(seed)
        self.w_penalty = 1e5

    def decode(self, perm: List[int], rots: List[bool],
               width: float) -> Tuple[Dict[str, Tuple[float, float, bool]], float]:
        """Pack modules into shelves with a fixed row width."""
        positions = {}
        x = 0.0
        y = 0.0
        row_h = 0.0
        for idx in perm:
            module = self.modules[idx]
            w, h = self._size(module, rots[idx])
            if x > 0.0 and x + w > width + 1e-9:
                y += row_h
                x = 0.0
                row_h = 0.0
            positions[module.name] = (x, y, rots[idx])
            x += w
            row_h = max(row_h, h)
        return positions, y + row_h

    def hpwl(self, positions: Dict[str, Tuple[float, float, bool]]) -> float:
        """Compute HPWL using module centers and fixed terminal positions."""
        pin_positions = {}
        for module in self.modules:
            if module.name not in positions:
                continue
            x, y, rotated = positions[module.name]
            w, h = self._size(module, rotated)
            pin_positions[module.name] = (x + w / 2.0, y + h / 2.0)
        pin_positions.update(self.terminal_positions)

        total = 0.0
        for net in self.nets:
            xs = []
            ys = []
            for pin in net.pins:
                if pin in pin_positions:
                    px, py = pin_positions[pin]
                    xs.append(px)
                    ys.append(py)
            if xs:
                total += ((max(xs) - min(xs)) + (max(ys) - min(ys))) * net.weight
        return total

    def anneal(self, width: float, objective: str,
               max_total_iter: int = 20000,
               max_iter_per_temp: int = 80,
               t_final: float = 0.5,
               cooling_rate: float = 0.995,
               init: Optional[Tuple[List[int], List[bool]]] = None,
               strict: bool = False):
        """Optimize either fixed-outline feasibility or HPWL."""
        if init is None:
            perm, rots = self._initial(width)
        else:
            perm = list(init[0])
            rots = list(init[1])

        current = self._cost(perm, rots, width, objective)
        temperature = max(2000.0, current / 6.0)
        best = (current, list(perm), list(rots))
        recent = []
        iteration = 0

        while temperature > t_final and iteration < max_total_iter:
            for _ in range(max_iter_per_temp):
                if iteration >= max_total_iter:
                    break
                iteration += 1
                next_perm, next_rots = self._perturb(perm, rots)
                if strict and objective == "hpwl":
                    _, next_h = self.decode(next_perm, next_rots, width)
                    if next_h > width + 1e-6:
                        continue

                next_cost = self._cost(next_perm, next_rots, width, objective)
                delta = next_cost - current
                if delta < 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                    perm = next_perm
                    rots = next_rots
                    current = next_cost
                    if current < best[0]:
                        best = (current, list(perm), list(rots))

                _, height = self.decode(perm, rots, width)
                recent.append(1 if height > width + 1e-9 else 0)
                if len(recent) > 200:
                    recent.pop(0)
                if len(recent) >= 50:
                    rate = sum(recent) / len(recent)
                    if rate > 0.5:
                        self.w_penalty = min(self.w_penalty * 1.12, 1e9)
                    elif rate < 0.06:
                        self.w_penalty = max(self.w_penalty * 0.9, 10.0)
            temperature *= cooling_rate

        _, best_perm, best_rots = best
        positions, height = self.decode(best_perm, best_rots, width)
        return positions, height, best_perm, best_rots

    def _cost(self, perm: List[int], rots: List[bool],
              width: float, objective: str) -> float:
        positions, height = self.decode(perm, rots, width)
        violation = max(0.0, height - width) / max(width, 1e-9)
        if objective == "hpwl":
            return self.hpwl(positions) + violation * self.w_penalty
        return violation * self.w_penalty

    def _initial(self, width: float, best_of: int = 18):
        candidates = []
        sorters = (
            lambda i: -self.modules[i].height,
            lambda i: -self.modules[i].width,
            lambda i: -self.modules[i].area,
        )
        for sorter in sorters:
            order = sorted(range(self.n), key=sorter)
            candidates.append((order, [False] * self.n))
            candidates.append((order, [True] * self.n))
        for _ in range(max(0, best_of - len(candidates))):
            order = list(range(self.n))
            self.rng.shuffle(order)
            candidates.append((order, [self.rng.random() < 0.5 for _ in range(self.n)]))

        best = None
        for order, rots in candidates:
            _, height = self.decode(order, rots, width)
            if best is None or height < best[0]:
                best = (height, list(order), list(rots))
        return best[1], best[2]

    def _perturb(self, perm: List[int], rots: List[bool]):
        next_perm = list(perm)
        next_rots = list(rots)
        move = self.rng.randrange(3)
        if move == 0:
            a = self.rng.randrange(self.n)
            b = self.rng.randrange(self.n)
            if a != b:
                next_perm[a], next_perm[b] = next_perm[b], next_perm[a]
        elif move == 1:
            a = self.rng.randrange(self.n)
            b = self.rng.randrange(self.n)
            if a != b:
                value = next_perm.pop(a)
                next_perm.insert(b, value)
        else:
            a = self.rng.randrange(self.n)
            next_rots[a] = not next_rots[a]
        return next_perm, next_rots

    @staticmethod
    def _size(module: Module, rotated: bool) -> Tuple[float, float]:
        if rotated and module.is_hard:
            return module.height, module.width
        return module.width, module.height
