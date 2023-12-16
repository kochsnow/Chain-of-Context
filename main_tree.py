import json
import os
import pickle

from eval.tasks.proofwriter import ProofWriterBase

from eval.tasks import get_task
from tree_node.executor import execute_tree
from tree_node.TranslationPassTreeNode import TranslationPassTreeNode

task_name = "proofwriter-neurosymbolic-2shot"
task = ProofWriterBase(mode='neurosymbolic', n=2)

dataset = task.get_dataset()

nodes = []
n = 60
for i in range(n):
    nodes.append(
        TranslationPassTreeNode(dataset[i], task_name, task, model='gpt-4-0613', chat=True)
    )

# if os.path.exists('temp_pickle.pkl'):
#     with open("temp_pickle.pkl", 'rb') as f:
#         nodes = pickle.load(f)


def pickle_nodes():
    print("saving...")
    with open("temp_pickle.pkl", 'wb') as f:
        pickle.dump(nodes, f)


execute_tree(nodes, save_callback=pickle_nodes)

for i, node in enumerate(nodes):
    print(i)
    print(node.get_children_votes())

print('-' * 30)

for i in range(n):
    print(task.get_reference(dataset[i]))
    print(nodes[i].get_children_votes())

collected_result = []
final_results = []
for i in range(n):
    votes = nodes[i].get_children_votes()
    mx = max(votes.values())
    final_ans = [k for k in votes if votes[k] == mx][0]
    collected_result.append(votes)
    final_results.append(final_ans)


with open('temp_dataset_output_all.json', "w") as fp:
    json.dump(collected_result, fp)

with open('temp_dataset_output_answers.json', "w") as fp:
    json.dump(collected_result, fp)
