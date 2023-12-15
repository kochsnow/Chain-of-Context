import xml.etree.ElementTree as ET

from dataclasses import dataclass
from typing import List


@dataclass
class SentenceList:
    sentences: List["Sentence"]

    def format(self):
        new_line = '\n'
        return new_line.join([p.format() for p in self.sentences])

    @staticmethod
    def from_xml(root):
        return SentenceList(
            sentences=[Sentence.from_xml(p) for p in root.findall("SENTENCE")]
        )


@dataclass
class AnnotatedSentenceList:
    sentences: List["AnnotatedSentence"]

    def format(self):
        new_line = '\n'
        return new_line.join([p.format() for p in self.sentences])

    @staticmethod
    def from_xml(root):
        return SentenceList(
            sentences=[Sentence.from_xml(p) for p in root.findall("ANNOTATED_SENTENCE")]
        )


@dataclass
class RawPremiseAndConclusion:
    premises: List[str]
    conclusion: str

    def format(self):
        new_line = '\n'
        return f"""
<RAW_PREMISE_AND_CONCLUSION>
    {new_line.join(f'<PREMISE>{p}</PREMISE>' for p in self.premises)}
    <CONCLUSION>{self.conclusion}</CONCLUSION>
</RAW_PREMISE_AND_CONCLUSION>
"""

    @staticmethod
    def from_xml(root):
        return RawPremiseAndConclusion(
            premises=[p.text for p in root.findall("PREMISE").text],
            conclusion=root.find("CONCLUSION").text
        )


@dataclass
class Sentence:
    premise: str
    fol: str

    def format(self):
        return f"""
<SENTENCE>
    <PREMISE>{self.premise}</PREMISE>
    <FOL>{self.fol}</FOL>
</SENTENCE>
"""

    @staticmethod
    def from_xml(root):
        return Sentence(
            premise=root.find("PREMISE").text,
            fol=root.find("FOL").text
        )


@dataclass
class ContextSentence:
    context: str
    fol: str
    justification: str

    def format(self):
        return f"""
<CONTEXT_SENTENCE>
    <CONTEXT>{self.context}</CONTEXT>
    <FOL>{self.fol}</FOL>
    <JUSTIFICATION>{self.justification}</JUSTIFICATION>
</CONTEXT_SENTENCE>
"""

    @staticmethod
    def from_xml(root):
        return ContextSentence(
            context=root.find("CONTEXT").text,
            fol=root.find("FOL").text,
            justification=root.find("JUSTIFICATION").text
        )


@dataclass
class Translation:
    premises: List[Sentence]
    conclusion: Sentence

    def format(self):
        newline = '\n'
        return f"""
<TRANSLATION>
    <PREMISES>
    {newline.join([p.format() for p in self.premises])}
    </PREMISES>
    <CONCLUSION>
    {self.conclusion.format()}
    </CONCLUSION>
</TRANSLATION>
"""

    @staticmethod
    def from_xml(root):
        premises = root.find("PREMISES")
        conclusion = root.find("CONCLUSION")
        return Translation(
            premises=[Sentence.from_xml(p) for p in premises],
            conclusion=Sentence.from_xml(conclusion)
        )


@dataclass
class AnnotatedSentence:
    sentence: Sentence
    contexts: List[ContextSentence]

    def format(self):
        newline = '\n'
        return f"""
    <ANNOTATED_SENTENCE>
        {self.sentence.format()}
        {newline.join([c.format() for c in self.contexts])}
    </ANNOTATED_SENTENCE>
"""

    @staticmethod
    def from_xml(root):
        sentence = root.find("SENTENCE")
        contexts = root.findall("CONTEXT_SENTENCE")
        return AnnotatedSentence(
            sentence=Sentence.from_xml(sentence),
            contexts=[ContextSentence.from_xml(c) for c in contexts]
        )


@dataclass
class AnnotatedTranslation:
    annotated_premises: List[AnnotatedSentence]
    conclusion: Sentence

    def format(self):
        newline = '\n'
        return f"""
    <ANNOTATED_TRANSLATION>
        <ANNOTATED_PREMISES>
        {newline.join(AnnotatedSentence.format(p) for p in self.annotated_premises)}
        </ANNOTATED_PREMISES>
        <CONCLUSION>
        {self.conclusion.format()}
        </CONCLUSION>
    </ANNOTATED_TRANSLATION>
    """

    @staticmethod
    def from_xml(root):
        return AnnotatedTranslation(
            annotated_premises=[AnnotatedSentence.from_xml(p) for p in root.find("ANNOTATED_PREMISES").findall("ANNOTATED_SENTENCE")],
            conclusion=Sentence.from_xml(root.find("CONCLUSION"))
        )
