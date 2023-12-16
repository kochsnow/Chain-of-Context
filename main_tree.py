from eval.tasks import get_task
from tree_node.executor import execute_tree
from tree_node.TranslationPassTreeNode import TranslationPassTreeNode

task_name = "proofwriter-neurosymbolic-2shot"
task = get_task(task_name)

doc = task.get_dataset()[3]
nodes = []
for i in range(10):
    nodes.append(
        TranslationPassTreeNode(doc, task_name, task, model='gpt-3.5-turbo', chat=True)
    )
execute_tree(nodes)

for i, node in enumerate(nodes):
    print(i)
    print(node.get_children_votes())

breakpoint()
