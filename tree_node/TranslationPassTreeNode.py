import json
import os

from ContextPassTreeNode import ContextPassTreeNode
from llm.LLM import OAILLM
from TreeNode import TreeNode
from eval.base import OWAFOLTask
from settings import CACHE_DIR, OPENAI_API_ENV_KEYS, TOP_P, MAX_LENGTH_GENERATION, TEMPERATURE, \
    SYSTEM_CHAT_INSTRUCTION, TRANSLATION_PASS_N_SAMPLES, get_next_recent_query_counter, RECENT_QUERY_DIR
from tree_node.datatypes import Translation, to_sentence


class TranslationPassTreeNode(TreeNode):
    def __init__(self, doc, task_name, task: OWAFOLTask, model, chat):
        super(TranslationPassTreeNode, self).__init__(doc, model, task_name, task)
        self.stop_words = task.stop_words
        self.stop_words.append("</OUTPUT>")
        self.chat = chat
        self.llm = OAILLM(
            chat=chat,
            model=model,
            n_samples=TRANSLATION_PASS_N_SAMPLES,
            cache_dir=CACHE_DIR,
            openai_api_env_keys=OPENAI_API_ENV_KEYS,
            chat_system_instruction=SYSTEM_CHAT_INSTRUCTION,
            top_p=TOP_P,
            max_length_generation=MAX_LENGTH_GENERATION,
            temperature=TEMPERATURE
        )

    def name(self):
        return f"{self.model}_{self.task_name}.json"

    def get_prompt(self):
        instructions = self.task.get_instructions()
        train = self.task.fewshot_examples()
        test = self.task.format_test_example(self.doc)
        prompt = "\n".join([instructions, train, test])
        return prompt

    def expand(self):
        prompt = self.get_prompt()
        stop = self.stop_words()
        response = self.llm.get_completion(prompt=prompt, stop=stop)
        filename = os.path.join(RECENT_QUERY_DIR, str(get_next_recent_query_counter()))
        with open(filename, "w") as fp:
            data = {"prompt": prompt, "stop": stop, "response": response}
            json.dump(data, fp)
        for resp in response:
            lines = resp.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            translation = Translation(
                premises=[to_sentence(lines[0], lines[1]) for i in range(0, len(lines)-2, 2)],
                conclusion=to_sentence(lines[-2], lines[-1])
            )
            self.children.append(
                ContextPassTreeNode(
                    doc=self.doc, translation=translation, task_name=self.task_name, task=self.task,
                    model=self.model, chat=self.chat
                )
            )
        return self.children
