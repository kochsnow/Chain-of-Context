from typing import List
from .TreeNode import TreeNode
from concurrent.futures import ThreadPoolExecutor


def expand_node(node: TreeNode):
    result = node.expand()
    return result


def execute_tree(nodes: List[TreeNode]):
    while len(nodes) > 0:
        with ThreadPoolExecutor() as executor:
            new_children = executor.map(expand_node, nodes)
            nodes = []
            for children in new_children:
                nodes.extend(children)
