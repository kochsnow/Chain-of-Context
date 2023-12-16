from typing import List
from .TreeNode import TreeNode
from concurrent.futures import ThreadPoolExecutor


def expand_node(node: TreeNode):
    result = node.expand()
    return result


def execute_tree(nodes: List[TreeNode], save_callback=None):
    while len(nodes) > 0:
        print(len(nodes), 'nodes in the queue')
        with ThreadPoolExecutor() as executor:
            new_children = executor.map(expand_node, nodes)
            nodes = []
            for children in new_children:
                nodes.extend(children)
            save_callback()


def find_all_leafs(nodes):
    leafs = []
    while len(nodes) > 0:
        next_layer = []
        for node in nodes:
            if len(node.children) == 0:
                leafs.append(node)
            else:
                next_layer.extend(node.children)
        nodes = next_layer
    return leafs
