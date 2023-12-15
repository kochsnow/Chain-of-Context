import json
import os

import xml.etree.ElementTree as ET

from llm.LLM import OAILLM
from TreeNode import TreeNode
from eval.base import OWAFOLTask
from settings import CACHE_DIR, OPENAI_API_ENV_KEYS, TOP_P, MAX_LENGTH_GENERATION, TEMPERATURE, \
    SYSTEM_CHAT_INSTRUCTION, CONTEXT_PASS_N_SAMPLES, get_next_recent_query_counter, RECENT_QUERY_DIR
from .ResultPassTreeNode import ResultTreeNode
from .datatypes import Translation, AnnotatedTranslation, ContextSentence, Sentence, AnnotatedSentence, SentenceList, \
    AnnotatedSentenceList


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
{example_input_1.format()}
</INPUT>
<OUTPUT>
{example_output_1.format()}
</OUTPUT>

<INPUT>
{SentenceList(sentences=self.translation.premises).format()}
</INPUT>
<OUTPUT>
    """

    def expand(self):
        self.response = self.make_request_and_log(llm=self.llm, prompt=self.get_prompt(), stop=self.stop_words)
        for resp in self.response:
            annotated_premises = AnnotatedSentenceList.from_xml(ET.fromstring(resp))
            annotated_translation = AnnotatedTranslation(annotated_premises=annotated_premises, conclusion=self.translation.conclusion)
            self.children.append(
                ResultTreeNode(
                    doc=self.doc, annotated_translation=annotated_translation, task_name=self.task_name, task=self.task,
                    model=self.model, chat=self.chat
                )
            )
        return self.children


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

example_input_1 = SentenceList.from_xml(ET.fromstring("""
<SENTENCE>
    PREMISE: When a person reads a book, that person gains knowledge.
    FOL: all x. all y. (Person(x) & Reads(x, y) & Book(y) -> Gains(x, Knowledge))
</SENTENCE>
<SENTENCE>
    <PREMISE> Harry read the book "Walden" by Henry. </PREMISE>
    <FOL> Reads(Harry, Walden) </FOL>
</SENTENCE>
        """))

example_output_1 = AnnotatedSentenceList.from_xml(ET.fromstring("""
<ANNOTATED_SENTENCE>
    <PREMISE> When a person reads a book, that person gains knowledge. </PREMISE>
    <FOL> all x. all y. (Person(x) & Reads(x, y) & Book(y) -> Gains(x, Knowledge)) </FOL>
</ANNOTATED_SENTENCE>
<ANNOTATED_SENTENCE>
    <PREMISE> Harry read the book "Walden" by Henry. </PREMISE>
    <FOL> Reads(Harry, Walden) </FOL>
    <CONTEXT_SENTENCE>
        <CONTEXT> Harry is a person. </CONTEXT>
        <FOL> Person(Harry) </FOL>
        <JUSTIFICATION> Harry is a person's name - this categorizes proper nouns. </JUSTIFICATION>
    </CONTEXT_SENTENCE>
    <CONTEXT_SENTENCE>
        <CONTEXT> Walden is a book. </CONTEXT> 
        <FOL> Book(Walden) </FOL>
        <JUSTIFICATION> The text says 'the book "Walden"', so Walden is a book's name - this categorizes proper nouns.  </JUSTIFICATION>
    </CONTEXT_SENTENCE>
</ANNOTATED_SENTENCE>
"""))


# todo
"""
<INPUT>
    <SENTENCE>
        PREMISE: Heinrich Schmidt was a Nazi German politician.
        FOL: NaziGermanPolitician(HeinrichSchmidt)
    </SENTENCE>
</INPUT>

<OUTPUT>
    <SENTENCE>
        PREMISE: Heinrich Schmidt was a Nazi German politician.
        FOL: NaziGermanPolitician(HeinrichSchmidt)
        <CONTEXT_SENTENCE>
            CONTEXT: Heinrich Schmidt was a Nazi.
            FOL: Nazi(HeinrichSchmidt)
            JUSTIFICATION: Because Heinrich Schmidt was a Nazi German politician, he must have been Nazi. 
        </CONTEXT_SENTENCE>
        <CONTEXT_SENTENCE>
            CONTEXT: Heinrich Schmidt was a German.
            FOL: German(HeinrichSchmidt)
            JUSTIFICATION: Because Heinrich Schmidt was a Nazi German politician, he must have been German. 
        </CONTEXT_SENTENCE>
        <CONTEXT_SENTENCE>
            CONTEXT: Heinrich Schmidt was a Politician.
            FOL: Politician(HeinrichSchmidt)
            JUSTIFICATION: Because Heinrich Schmidt was a Nazi German politician, he must have been a politician. 
        </CONTEXT_SENTENCE>
    </SENTENCE>
</OUTPUT>

<INPUT>
    <SENTENCE>
        PREMISE: Famine is bad.
        FOL: Bad(Famine)
    </SENTENCE>
</INPUT>
<OUTPUT>
    <SENTENCE>
        PREMISE: Famine is bad.
        FOL: Bad(Famine)
        <CONTEXT_SENTENCE>
            CONTEXT: Bad is not good.
            FOL: all x. Bad(x) -> -Good(x)
            JUSTIFICATION: The adds context that describes the nature of antonyms.
        </CONTEXT_SENTENCE>
    </SENTENCE>
</OUTPUT>
"""