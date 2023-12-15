from typing import List
from .TreeNode import TreeNode
from concurrent.futures import ThreadPoolExecutor


def expand_node(node: TreeNode):
    return node.expand()


def execute_tree(nodes: List[TreeNode]):
    while len(nodes) > 0:
        with ThreadPoolExecutor() as executor:
            new_children = executor.map(expand_node, nodes)
            nodes = [node for node in lst for lst in new_children]
