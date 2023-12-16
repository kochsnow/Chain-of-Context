import json
import os

from llm.LLM import OAILLM
from .TreeNode import TreeNode
from eval.base import OWAFOLTask
from settings import CACHE_DIR, OPENAI_API_ENV_KEYS, TOP_P, MAX_LENGTH_GENERATION, TEMPERATURE, \
    SYSTEM_CHAT_INSTRUCTION, CONTEXT_PASS_N_SAMPLES, get_next_recent_query_counter, RECENT_QUERY_DIR
from .ResultPassTreeNode import ResultTreeNode
from .datatypes import Translation, AnnotatedTranslation, ContextSentence, Sentence, AnnotatedSentence, SentenceList, \
    AnnotatedSentenceList, AnnotatedSentence


class ContextPassTreeNode(TreeNode):
    def __init__(self, doc, translation: Translation, task_name, task: OWAFOLTask, model, chat):
        super(ContextPassTreeNode, self).__init__(doc, model, task_name, task)
        self.stop_words = task.stop_words
        self.stop_words.append("</OUTPUT>")
        self.translation = translation
        self.chat = chat
        self.llm = OAILLM(
            chat=chat,
            model=model,
            n_samples=CONTEXT_PASS_N_SAMPLES,
            cache_dir=CACHE_DIR,
            openai_api_env_keys=OPENAI_API_ENV_KEYS,
            chat_system_instruction=SYSTEM_CHAT_INSTRUCTION,
            top_p=TOP_P,
            max_length_generation=MAX_LENGTH_GENERATION,
            temperature=TEMPERATURE
        )
        self.response = None

    def get_prompt(self):
        return f"""
{CONTEXT_PROMPT_INSTRUCTION}

<INPUT>
{example_input_1.to_string()}
</INPUT>
<OUTPUT>
{example_output_1.to_string()}
</OUTPUT>

<INPUT>
{SentenceList(sentences=self.translation.premises).to_string()}
</INPUT>
<OUTPUT>
    """

    def expand(self):
        self.response = self.make_request_and_log(llm=self.llm, prompt=self.get_prompt(), stop=self.stop_words)
        for resp in self.response:
            annotated_premises = AnnotatedSentenceList.from_string(resp).sentences
            annotated_translation = AnnotatedTranslation(annotated_premises=annotated_premises, conclusion=self.translation.conclusion)
            self.children.append(
                ResultTreeNode(
                    doc=self.doc, annotated_translation=annotated_translation, task_name=self.task_name, task=self.task,
                    model=self.model, chat=self.chat
                )
            )
        return self.children


    # # todo remove this hacky cache
    # def make_request_and_log(self, llm: LLM, prompt, stop):
    #     return ['{\n    "sentences": [\n        {\n            "sentence": {\n                "premise": "Charlie is cold.",\n                "fol": "Cold(Charlie)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Charlie is quiet.",\n                "fol": "Quiet(Charlie)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Dave is blue.",\n                "fol": "Blue(Dave)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Dave is furry.",\n                "fol": "Furry(Dave)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Dave is nice.",\n                "fol": "Nice(Dave)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Dave is quiet.",\n                "fol": "Quiet(Dave)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Fiona is furry.",\n                "fol": "Furry(Fiona)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Fiona is quiet.",\n                "fol": "Quiet(Fiona)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Fiona is red.",\n                "fol": "Red(Fiona)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Fiona is smart.",\n                "fol": "Smart(Fiona)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Harry is cold.",\n                "fol": "Cold(Harry)"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "All blue things are red.",\n                "fol": "all x. (Blue(x) -> Red(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Blue, nice things are quiet.",\n                "fol": "all x. ((Blue(x) & Nice(x)) -> Quiet(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "If Harry is quiet then Harry is furry.",\n                "fol": "all x. (Quiet(x) -> Furry(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "If Charlie is smart and Charlie is cold then Charlie is furry.",\n                "fol": "all x. (Smart(x) & Cold(x) -> Furry(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "If something is furry then it is nice.",\n                "fol": "all x. (Furry(x) -> Nice(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Red, quiet things are smart.",\n                "fol": "all x. ((Red(x) & Quiet(x)) -> Smart(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Nice things are smart.",\n                "fol": "all x. (Nice(x) -> Smart(x))"\n            },\n            "contexts": []\n        },\n        {\n            "sentence": {\n                "premise": "Quiet things are blue.",\n                "fol": "all x. (Quiet(x) -> Blue(x))"\n            },\n            "contexts": []\n        }\n    ]\n}']


CONTEXT_PROMPT_INSTRUCTION = f"""You will be given the premises for a first-order logic (FOL) problem.
The problem is to identify additional premises that are implicitly common sense from the ones given, and label them.
The original premises are given in the form of a set of premises and first-order logic sentences.
The task is to generate new common sense premises, text and FOL pairs, that would be common sense to someone reading the original premises. Include this new context in the sentence that it describes, with the correct label of CONTEXT, and provide brief justification.
These new common sense, context-based premises should reflect context: the nature of synonyms and antonyms, categorize proper names, and identify implicit characteristics from the ones provided.
For the new context, provide CONTEXT in text, FOL in the format of the Python NLTK, and a brief justification. 
Do not create new sentences for context, instead, add the context to the sentence it fits best in. 
Only add context to a sentence based on the premise and real-world common-sense, and do not make any inferences using other sentences. 
Do not limit the amount of new premises generated in the output.
Expressions should be adhere to the format of the Python NLTK package logic module. Here are a couple examples:
"""

example_input_1 = SentenceList(
    sentences=[
        Sentence(
            premise="When a person reads a book, that person gains knowledge.",
            fol="all x. all y. (Person(x) & Reads(x, y) & Book(y) -> Gains(x, Knowledge))"
        ),
        Sentence(
            premise='Harry read the book "Walden" by Henry.',
            fol='Reads(Harry, Walden)'
        )
    ]
)

example_output_1 = AnnotatedSentenceList(
    sentences=[
        AnnotatedSentence(
            Sentence(
                premise="When a person reads a book, that person gains knowledge.",
                fol="all x. all y. (Person(x) & Reads(x, y) & Book(y) -> Gains(x, Knowledge))"
            ),
            contexts=[]
        ),
        AnnotatedSentence(
            Sentence(
                premise='Harry read the book "Walden" by Henry.',
                fol='Reads(Harry, Walden)'
            ),
            contexts=[
                ContextSentence(
                    context="Harry is a person.",
                    fol="Person(Harry)",
                    justification="Harry is a person's name - this categorizes proper nouns."
                ),
                ContextSentence(
                    context="Walden is a book.",
                    fol="Book(Walden)",
                    justification="""The text says 'the book "Walden"', so Walden is a book's name - this categorizes proper nouns."""
                )
            ]
        )
    ]
)

# todo add this
"""
<ANNOTATED_SENTENCE>
    <PREMISE> Heinrich Schmidt was a Nazi German politician. </PREMISE>
    <FOL> NaziGermanPolitician(HeinrichSchmidt) </FOL>
</ANNOTATED_SENTENCE>
<ANNOTATED_SENTENCE>
    <CONTEXT_SENTENCE>
        <CONTEXT> Heinrich Schmidt was a Nazi. </CONTEXT>
        <FOL> Nazi(HeinrichSchmidt) </FOL>
        <JUSTIFICATION> Because Heinrich Schmidt was a Nazi German politician, he must have been a Nazi. </JUSTIFICATION>
    </CONTEXT_SENTENCE>
    <CONTEXT_SENTENCE>
        <CONTEXT> Heinrich Schmidt was German. </CONTEXT>
        <FOL> German(HeinrichSchmidt) </FOL>
        <JUSTIFICATION> Because Heinrich Schmidt was a Nazi German politician, he must have been German. </JUSTIFICATION>
    </CONTEXT_SENTENCE>
    <CONTEXT_SENTENCE>
        <CONTEXT> Heinrich Schmidt was a Politician. </CONTEXT>
        <FOL> Politician(HeinrichSchmidt) </FOL>
        <JUSTIFICATION> Because Heinrich Schmidt was a Nazi German politician, he must have been a politician. </JUSTIFICATION>
    </CONTEXT_SENTENCE>
</ANNOTATED_SENTENCE>


<ANNOTATED_SENTENCE>
    <PREMISE> Famine is bad. </PREMISE>
    <FOL> Bad(Famine) </FOL>
</ANNOTATED_SENTENCE>
<ANNOTATED_SENTENCE>
    <CONTEXT_SENTENCE>
        <CONTEXT> Bad is not good. </CONTEXT>
        <FOL> all x. Bad(x) -> -Good(x) </FOL>
        <JUSTIFICATION> The context adds an understanding of the nature of antonyms. </JUSTIFICATION>
    </CONTEXT_SENTENCE>
</ANNOTATED_SENTENCE>

"""