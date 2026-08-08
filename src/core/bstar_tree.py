# 本文件实现 B*-Tree 表示法和基于轮廓线的矩形模块 packing，
# 用于把模拟退火中的树结构快速转换为合法的不重叠布局坐标。
"""VLSI floorplanning solver modules split from the original spr backup."""


import random
from typing import Dict, List, Optional, Tuple

from .models import Module

class BTreeNode:
    """Node in a B*-tree."""
    __slots__ = ('module_idx', 'left', 'right', 'parent')

    def __init__(self, module_idx: int):
        self.module_idx = module_idx
        self.left = None   # placed to the right in floorplan
        self.right = None  # placed above in floorplan
        self.parent = None



class BTree:
    """
    B*-tree representation for non-slicing floorplans.

    The B*-tree encodes a compacted floorplan:
    - Root node is placed at bottom-left
    - Left child: placed to the right of parent
    - Right child: placed above parent
    - Contour-based packing computes actual coordinates
    """

    def __init__(self, num_modules: int):
        self.num_modules = num_modules
        self.root = None
        self.nodes: List[Optional[BTreeNode]] = [None] * num_modules

    def copy(self) -> 'BTree':
        """Deep copy the tree."""
        new_tree = BTree(self.num_modules)
        if self.root:
            new_tree.root = new_tree._copy_node(self.root, None, new_tree)
        return new_tree

    def _copy_node(self, node: BTreeNode, parent: Optional[BTreeNode],
                   new_tree: 'BTree') -> BTreeNode:
        new_node = BTreeNode(node.module_idx)
        new_node.parent = parent
        new_tree.nodes[node.module_idx] = new_node
        if node.left:
            new_node.left = self._copy_node(node.left, new_node, new_tree)
        if node.right:
            new_node.right = self._copy_node(node.right, new_node, new_tree)
        return new_node

    def build_initial_tree(self, module_indices: List[int]):
        """Build an initial balanced-ish B*-tree from a list of module indices."""
        self.root = None
        self.nodes = [None] * self.num_modules
        if not module_indices:
            return
        self.root = self._build_balanced(module_indices, 0, len(module_indices) - 1, None)

    def build_left_chain(self, module_indices: List[int]):
        """构造左孩子链：后续模块不断放到当前模块右侧。"""
        self._build_chain(module_indices, use_left_child=True)

    def build_right_chain(self, module_indices: List[int]):
        """构造右孩子链：后续模块不断放到当前模块上方。"""
        self._build_chain(module_indices, use_left_child=False)

    def _build_chain(self, module_indices: List[int], use_left_child: bool):
        """按给定顺序构造单链 B*-Tree。"""
        self.root = None
        self.nodes = [None] * self.num_modules
        prev = None
        for module_idx in module_indices:
            node = BTreeNode(module_idx)
            self.nodes[module_idx] = node
            if prev is None:
                self.root = node
            else:
                node.parent = prev
                if use_left_child:
                    prev.left = node
                else:
                    prev.right = node
            prev = node

    def _build_balanced(self, indices: List[int], start: int, end: int,
                        parent: Optional[BTreeNode]) -> Optional[BTreeNode]:
        if start > end:
            return None
        mid = (start + end) // 2
        node = BTreeNode(indices[mid])
        node.parent = parent
        self.nodes[indices[mid]] = node
        node.left = self._build_balanced(indices, mid + 1, end, node)
        node.right = self._build_balanced(indices, start, mid - 1, node)
        return node

    def get_all_nodes(self) -> List[BTreeNode]:
        """Return all nodes in the tree (in-order traversal)."""
        nodes = []
        self._inorder(self.root, nodes)
        return nodes

    def _inorder(self, node: Optional[BTreeNode], result: List[BTreeNode]):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node)
        self._inorder(node.right, result)

    def get_node(self, module_idx: int) -> Optional[BTreeNode]:
        """Find a node by module index."""
        if 0 <= module_idx < len(self.nodes):
            return self.nodes[module_idx]
        return None

    def _find(self, node: Optional[BTreeNode], module_idx: int) -> Optional[BTreeNode]:
        if node is None:
            return None
        if node.module_idx == module_idx:
            return node
        left_result = self._find(node.left, module_idx)
        if left_result:
            return left_result
        return self._find(node.right, module_idx)

    def swap_modules(self, idx1: int, idx2: int):
        """Swap two modules in the B*-tree by swapping their indices in nodes."""
        node1 = self.get_node(idx1)
        node2 = self.get_node(idx2)
        if node1 and node2:
            node1.module_idx, node2.module_idx = node2.module_idx, node1.module_idx
            self.nodes[idx1], self.nodes[idx2] = node2, node1

    def is_ancestor(self, ancestor: BTreeNode, node: BTreeNode) -> bool:
        """判断 ancestor 是否为 node 的祖先节点。"""
        curr = node
        while curr is not None:
            if curr == ancestor:
                return True
            curr = curr.parent
        return False

    def swap_children(self, module_idx: int) -> bool:
        """交换某个节点的左右子树，用于改变局部 B*-Tree 几何关系。"""
        node = self.get_node(module_idx)
        if node is None or (node.left is None and node.right is None):
            return False
        node.left, node.right = node.right, node.left
        return True

    def move_subtree(self, move_idx: int, target_idx: int) -> bool:
        """
        将以 move_idx 为根的整棵子树移动到 target_idx 附近。

        中文说明：单节点移动只能改变一个模块的位置；子树移动可以整体重组
        一片局部布局，是问题一面积搜索中跳出局部最优的重要扰动。
        """
        move_node = self.get_node(move_idx)
        target_node = self.get_node(target_idx)
        if (
            move_node is None
            or target_node is None
            or move_node == target_node
            or move_node.parent is None
            or self.is_ancestor(move_node, target_node)
        ):
            return False

        # 中文说明：先从原父节点断开整棵子树，子树内部结构保持不变。
        old_parent = move_node.parent
        if old_parent.left == move_node:
            old_parent.left = None
        elif old_parent.right == move_node:
            old_parent.right = None
        move_node.parent = None

        # 中文说明：随机作为目标节点的左/右孩子插入；被挤下来的原子树
        # 挂到移动子树中的第一个空孩子位置，避免节点丢失。
        if random.random() < 0.5:
            displaced = target_node.left
            target_node.left = move_node
        else:
            displaced = target_node.right
            target_node.right = move_node
        move_node.parent = target_node

        if displaced is not None:
            self._attach_to_first_empty(move_node, displaced)
        return True

    def _attach_to_first_empty(self, root: BTreeNode, subtree: BTreeNode):
        """把 subtree 挂到 root 子树中的第一个空孩子位置。"""
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left is None:
                node.left = subtree
                subtree.parent = node
                return
            if node.right is None:
                node.right = subtree
                subtree.parent = node
                return
            stack.append(node.right)
            stack.append(node.left)

    def delete_and_insert(self, delete_idx: int, insert_idx: int):
        """
        Delete module delete_idx from tree and re-insert it near insert_idx.
        This is one of the SA perturbation operations.
        """
        node = self.get_node(delete_idx)
        if node is None or delete_idx == insert_idx:
            return

        # Store the module index to be moved
        moved_module = node.module_idx

        # Remove node from tree (replace with its children or delete)
        self._delete_node(node)
        self.nodes[moved_module] = None

        # Insert near target
        target_node = self.get_node(insert_idx)
        if target_node:
            # 中文说明：标准 B*-Tree 移动算子要把模块真正插到目标节点附近。
            # 旧代码在目标左右孩子都存在时，会退化成“插到第一个空位置”，
            # 这会让邻域搜索失去局部意义。这里改为：随机选择左/右孩子位置，
            # 若该位置已有子树，则把原子树挂到新节点同侧，保证节点不丢失。
            new_node = BTreeNode(moved_module)
            new_node.parent = target_node
            self.nodes[moved_module] = new_node
            if random.random() < 0.5:
                old_child = target_node.left
                target_node.left = new_node
                new_node.left = old_child
            else:
                old_child = target_node.right
                target_node.right = new_node
                new_node.right = old_child
            if old_child:
                old_child.parent = new_node

    def _delete_node(self, node: BTreeNode):
        """Delete a node from the B*-tree, reattaching its children."""
        # Find replacement: prefer left child, then right child
        # Actually, for SA operations, we just remove and reinsert
        # A simpler approach: reconnect children
        replacement = node.left if node.left else node.right
        other_child = node.right if node.left else node.left

        if node.parent:
            if node.parent.left == node:
                node.parent.left = replacement
            else:
                node.parent.right = replacement
            if replacement:
                replacement.parent = node.parent
        else:
            self.root = replacement
            if replacement:
                replacement.parent = None

        # If there's an other_child, attach it to a suitable position
        if node.left and node.right:
            # Attach other_child to the rightmost position of replacement's left subtree
            if replacement:
                # other_child goes as far right as possible in replacement's subtree
                curr = replacement
                while curr.right:
                    curr = curr.right
                curr.right = other_child
                other_child.parent = curr


# ============================================================
# CONTOUR-BASED PACKING
# ============================================================


class ContourPacker:
    """
    Computes module positions from a B*-tree using contour-based packing.
    This guarantees a compacted (zero dead-space in y-direction for a given x-ordering)
    placement with no overlaps.
    """

    def __init__(self, modules: List[Module]):
        self.modules = modules

    def pack(self, tree: BTree) -> Tuple[Dict[int, Tuple[float, float]], float, float]:
        """
        Pack modules according to B*-tree.

        Returns:
            positions: dict mapping module_idx -> (x, y, rotated)
            total_width, total_height of the bounding box
        """
        if tree.root is None:
            return {}, 0, 0

        positions = {}
        contour = []  # List of (x_start, x_end, y_height) segments

        # Pre-order traversal: parent before children
        self._pack_node(tree.root, 0.0, contour, positions, tree)

        # Compute bounding box
        max_x = 0.0
        max_y = 0.0
        for idx, (x, y) in positions.items():
            mod = self.modules[idx]
            max_x = max(max_x, x + mod.w)
            max_y = max(max_y, y + mod.h)

        return positions, max_x, max_y

    def _pack_node(self, node: BTreeNode, parent_x: float,
                   contour: List[Tuple[float, float, float]],
                   positions: Dict[int, Tuple[float, float]],
                   tree: BTree):
        """Recursively pack a node and its children."""
        if node is None:
            return

        mod = self.modules[node.module_idx]
        w, h = mod.w, mod.h

        # Find y-coordinate using contour (skyline method)
        y = self._find_y(contour, parent_x, parent_x + w)

        positions[node.module_idx] = (parent_x, y)

        # Update contour
        self._update_contour(contour, parent_x, parent_x + w, y + h)

        # Process left child (placed to the right of current module)
        if node.left:
            self._pack_node(node.left, parent_x + w, contour, positions, tree)

        # Process right child (placed above current module, at same x as parent)
        if node.right:
            self._pack_node(node.right, parent_x, contour, positions, tree)

    def _find_y(self, contour: List[Tuple[float, float, float]],
                x_start: float, x_end: float) -> float:
        """Find the lowest y-position where a block of width (x_end-x_start) can be placed."""
        max_y = 0.0
        for seg_xs, seg_xe, seg_y in contour:
            # Check overlap
            if seg_xe > x_start and seg_xs < x_end:
                max_y = max(max_y, seg_y)
        return max_y

    def _update_contour(self, contour: List[Tuple[float, float, float]],
                        x_start: float, x_end: float, new_y: float):
        """Update the contour (skyline) after placing a block."""
        new_contour = []
        inserted = False
        for seg_xs, seg_xe, seg_y in contour:
            if seg_xe <= x_start or seg_xs >= x_end:
                if not inserted and seg_xs >= x_end:
                    new_contour.append((x_start, x_end, new_y))
                    inserted = True
                new_contour.append((seg_xs, seg_xe, seg_y))
            else:
                if seg_xs < x_start:
                    new_contour.append((seg_xs, x_start, seg_y))
                if not inserted:
                    new_contour.append((x_start, x_end, new_y))
                    inserted = True
                if seg_xe > x_end:
                    new_contour.append((x_end, seg_xe, seg_y))

        if not inserted:
            new_contour.append((x_start, x_end, new_y))

        merged = []
        for seg in new_contour:
            if merged and abs(merged[-1][1] - seg[0]) < 1e-9 and abs(merged[-1][2] - seg[2]) < 1e-9:
                # Merge
                merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
            else:
                merged.append(seg)
        contour[:] = merged


# ============================================================
# HPWL CALCULATION
# ============================================================
