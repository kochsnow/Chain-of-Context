import os
import json

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from llm import LLM
from settings import OUTPUTS_DIR, RECENT_QUERY_DIR, get_next_recent_query_counter
from typing import List


class TreeNode(ABC):
    def __init__(self, doc, model, task_name, task):
        self.doc = doc
        self.model = model
        self.task_name = task_name
        self.task = task
        self.children = []
        self.error = None

    @abstractmethod
    def expand(self) -> List["TreeNode"]:
        raise NotImplemented

    def name(self):
        return f"{self.model}_{self.task_name}.json"

    def make_request_and_log(self, llm: LLM, prompt, stop):
        print('making query')
        response = llm.get_completion(prompt=prompt, stop=stop)
        cnt = get_next_recent_query_counter()
        filenamejson = os.path.join(RECENT_QUERY_DIR, str(cnt) + ".json")
        filenametxt = os.path.join(RECENT_QUERY_DIR, str(cnt) + ".txt")
        with open(filenamejson, "w") as fp:
            data = {"prompt": prompt, "stop": stop, "response": response}
            json.dump(data, fp)
        with open(filenametxt, "w") as fp:
            fp.write("stop words:\n")
            fp.write(str(stop))
            fp.write('\n')
            fp.write("-" * 40)
            fp.write('\n')
            fp.write("prompt:\n")
            fp.write(str(prompt))
            fp.write('\n')
            fp.write("-" * 40)
            fp.write('\n')
            fp.write("response:\n")
            fp.write(str(response))
            print(response)
        print('finished query')
        return response

    def log(self, data):
        filename = os.path.join(OUTPUTS_DIR, self.name())
        with open(filename, "w") as fp:
            json.dump(data, fp)
            print(self.name(), 'saved')

    def get_children_votes(self):
        votes = {"True": 0, "False": 0, "Uncertain": 0, "Error": 0}
        if self.error:
            votes["Error"] = 1
            return votes
        for child in self.children:
            for vote, number in child.get_children_votes().items():
                votes[vote] += number
        return votes
