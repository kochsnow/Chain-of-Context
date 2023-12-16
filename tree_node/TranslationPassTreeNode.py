import json
import os

from .ContextPassTreeNode import ContextPassTreeNode
from llm.LLM import OAILLM
from .TreeNode import TreeNode
from eval.base import OWAFOLTask
from settings import CACHE_DIR, OPENAI_API_ENV_KEYS, TOP_P, MAX_LENGTH_GENERATION, TEMPERATURE, \
    SYSTEM_CHAT_INSTRUCTION, TRANSLATION_PASS_N_SAMPLES, get_next_recent_query_counter, RECENT_QUERY_DIR
from tree_node.datatypes import Translation, Sentence, RawPremiseAndConclusion


class TranslationPassTreeNode(TreeNode):
    def __init__(self, doc, task_name, task: OWAFOLTask, model, chat):
        super(TranslationPassTreeNode, self).__init__(doc, model, task_name, task)
        self.stop_words = task.stop_words
        self.stop_words.append('</OUTPUT>')

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
        self.response = None

    def name(self):
        return f"{self.model}_{self.task_name}.json"

    def get_prompt(self):
        instructions = self.task.get_instructions()

        n_shot = 2  # todo add this to config later
        formatted_examples = []
        for doc in self.task._train.select(range(n_shot)):
            premises = []
            premises_raw = []
            conclusion_raw = None
            conclusion = None
            for premise, fol in zip(doc['premises'], doc['premises-FOL']):
                premises.append(Sentence(premise=premise.strip(), fol=fol.strip()))
                premises_raw.append(premise.strip())
            conclusion = Sentence(premise=doc['conclusion'], fol=doc['conclusion-FOL'])
            conclusion_raw = doc['conclusion']
            formatted_examples.append(f'''
<INPUT>
{RawPremiseAndConclusion(premises=premises_raw, conclusion=conclusion_raw).to_string()}
</INPUT>
<OUTPUT>
{Translation(premises=premises, conclusion=conclusion).to_string()}
</OUTPUT>
            ''')

        test = RawPremiseAndConclusion(premises=[p for p in self.doc['premises']], conclusion=self.doc['conclusion'])
        newline = '\n'
        return f"""
{instructions}

{newline.join(formatted_examples)}

<INPUT>
{test.to_string()}
</INPUT>
<OUTPUT>
"""
    # def reformat_from_LINC(self, text):
    #     lines = text.split('\n')
    #     new_lines = []
    #     for line in lines:
    #         if ':' in line:
    #             sp = line.split(':')
    #             tag = sp[0]
    #             data = sp[1]
    #             assert len(sp) == 2
    #             line = f'<{tag}>{data}<{tag}/>'
    #         new_lines.append(lines)
    #     return '\n'.join(new_lines)

    def expand(self):
        self.response = self.make_request_and_log(llm=self.llm, prompt=self.get_prompt(), stop=self.stop_words)
        for resp in self.response:
            translation = Translation.from_string(resp)
            self.children.append(
                ContextPassTreeNode(
                    doc=self.doc, translation=translation, task_name=self.task_name, task=self.task,
                    model=self.model, chat=self.chat
                )
            )
        return self.children



    # # todo remove this hacky cache
    # def make_request_and_log(self, llm: LLM, prompt, stop):
    #     response = ['<PREMISES>\n\nPREMISE: Charlie is cold.\nFOL: Cold(Charlie)\n\n\nPREMISE: Charlie is quiet.\nFOL: Quiet(Charlie)\n\n\nPREMISE: Dave is blue.\nFOL: Blue(Dave)\n\n\nPREMISE: Dave is furry.\nFOL: Furry(Dave)\n\n\nPREMISE: Dave is nice.\nFOL: Nice(Dave)\n\n\nPREMISE: Dave is quiet.\nFOL: Quiet(Dave)\n\n\nPREMISE: Fiona is furry.\nFOL: Furry(Fiona)\n\n\nPREMISE: Fiona is quiet.\nFOL: Quiet(Fiona)\n\n\nPREMISE: Fiona is red.\nFOL: Red(Fiona)\n\n\nPREMISE: Fiona is smart.\nFOL: Smart(Fiona)\n\n\nPREMISE: Harry is cold.\nFOL: Cold(Harry)\n\n\nPREMISE: All blue things are red.\nFOL: all x. (Blue(x) -> Red(x))\n\n\nPREMISE: Blue, nice things are quiet.\nFOL: all x. (Blue(x) & Nice(x) -> Quiet(x))\n\n\nPREMISE: If Harry is quiet then Harry is furry.\nFOL: all x. (Quiet(x) -> Furry(x))\n\n\nPREMISE: If Charlie is smart and Charlie is cold then Charlie is furry.\nFOL: all x. (Smart(x) & Cold(x) -> Furry(x))\n\n\nPREMISE: If something is furry then it is nice.\nFOL: all x. (Furry(x) -> Nice(x))\n\n\nPREMISE: Red, quiet things are smart.\nFOL: all x. (Red(x) & Quiet(x) -> Smart(x))\n\n\nPREMISE: Nice things are smart.\nFOL: all x. (Nice(x) -> Smart(x))\n\n\nPREMISE: Quiet things are blue.\nFOL: all x. (Quiet(x) -> Blue(x))\n\n</PREMISES>\n<CONCLUSION>\n\nPREMISE: Charlie is smart.\nFOL: Smart(Charlie)\n\n<CONCLUSION>\n\n']
    #     return response
