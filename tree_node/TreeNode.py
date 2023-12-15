import os
import json

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from settings import OUTPUTS_DIR
from typing import List


class TreeNode(ABC):
    def __init__(self, doc, model, task_name, task):
        self.doc = doc
        self.model = model
        self.task_name = task_name
        self.task = task
        self.children = []

    @abstractmethod
    def expand(self) -> List["TreeNode"]:
        raise NotImplemented

    def name(self):
        return f"{self.model}_{self.task_name}.json"

    def log(self, data):
        filename = os.path.join(OUTPUTS_DIR, self.name())
        with open(filename, "w") as fp:
            json.dump(data, fp)
            print(self.name(), 'saved')
