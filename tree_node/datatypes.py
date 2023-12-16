import json
from dataclasses import dataclass
from typing import List
# import marshmallow_dataclass
# import marshmallow.validate


class JSONWizard:
    pass
    # def to_string(self):
    #     schema = marshmallow_dataclass.class_schema(self.__class__)()
    #     return schema.dump(self)
    #     # return json.dumps(self, default=lambda o: o.__dict__, indent=4)
    #
    # @classmethod
    # def from_string(cls, json_str):
    #     schema = marshmallow_dataclass.class_schema(cls)()
    #     json_str = json_str.replace('\'', '"')
    #     json_dict = json.loads(json_str)
    #     return schema.load(json_dict)
    #     # return cls(**json_dict)


@dataclass
class SentenceList(JSONWizard):
    sentences: List["Sentence"]

    def to_string(self):
        return "\n".join([
            f'''<SENTENCE>
{s.to_string()}
</SENTENCE>'''

            for s in self.sentences
        ])

    @staticmethod
    def from_string(text):
        return SentenceList(
            sentences=[Sentence.from_string(p) for p in get_blocks_of(text, "SENTENCE")]
        )


@dataclass
class AnnotatedSentenceList(JSONWizard):
    sentences: List["AnnotatedSentence"]

    def to_string(self):
        return "\n".join([
            f'''<ANNOTATED_SENTENCE>
{s.to_string()}
</ANNOTATED_SENTENCE>'''
            for s in self.sentences
        ])

    @staticmethod
    def from_string(text):
        return AnnotatedSentenceList(
            sentences=[AnnotatedSentence.from_string(p) for p in get_blocks_of(text, "ANNOTATED_SENTENCE")]
        )


@dataclass
class RawPremiseAndConclusion(JSONWizard):
    premises: List[str]
    conclusion: str

    def to_string(self):
        newline = '\n'
        return f"""
<PREMISES>
{newline.join(self.premises)}
</PREMISES>
<CONCLUSION>
{self.conclusion}
</CONCLUSION>
"""

    @staticmethod
    def from_string(text):
        premises_lines = get_blocks_of(text, "PREMISES")[0].split('\n')
        conclusion = get_blocks_of(text, "CONCLUSION")[0]
        return RawPremiseAndConclusion(
            premises=[line.strip() for line in premises_lines if line.strip()],
            conclusion=conclusion.strip()
        )


@dataclass
class Sentence(JSONWizard):
    premise: str
    fol: str

    def to_string(self):
        return f"""
PREMISE: {self.premise}
FOL: {self.fol}
"""

    @staticmethod
    def from_string(text):
        return Sentence(
            premise=get_variable(text, "PREMISE"),
            fol=get_variable(text, "FOL"),
        )


@dataclass
class ContextSentence(JSONWizard):
    context: str
    fol: str
    justification: str

    def to_string(self):
        return f"""
CONTEXT: {self.context}
FOL: {self.fol}
JUSTIFICATION: {self.justification}
"""

    @staticmethod
    def from_string(text):
        return ContextSentence(
            context=get_variable(text, "CONTEXT"),
            fol=get_variable(text, "FOL"),
            justification=get_variable(text, "JUSTIFICATION")
        )


@dataclass
class Translation(JSONWizard):
    premises: List[Sentence]
    conclusion: Sentence

    def to_string(self):
        newline = '\n'
        return f"""
<PREMISES>
{newline.join([p.to_string() for p in self.premises])}
</PREMISES>
<CONCLUSION>
{self.conclusion.to_string()}
<CONCLUSION>
"""

    @staticmethod
    def from_string(text):
        lines = get_blocks_of(text, "PREMISES")[0].split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        return Translation(
            premises=[Sentence.from_string(lines[i] + "\n" + lines[i+1]) for i in range(0, len(lines), 2)],
            conclusion=Sentence.from_string(get_blocks_of(text, "CONCLUSION")[0])
        )


@dataclass
class AnnotatedSentence(JSONWizard):
    sentence: Sentence
    contexts: List[ContextSentence]

    def to_string(self):
        newline = "\n"
        return f"""
<SENTENCE>
{self.sentence.to_string()}
</SENTENCE>
{newline.join([
f'''<EXTRA_CONTEXT>
{context.to_string()}
</EXTRA_CONTEXT>''' for context in self.contexts])}
"""

    @staticmethod
    def from_string(text):
        return AnnotatedSentence(
            sentence=Sentence.from_string(get_blocks_of(text, "SENTENCE")[0]),
            contexts=[ContextSentence.from_string(b) for b in get_blocks_of(text, "EXTRA_CONTEXT")]
        )


@dataclass
class AnnotatedTranslation(JSONWizard):
    annotated_premises: List[AnnotatedSentence]
    conclusion: Sentence

    def to_string(self):
        newline = '\n'
        return f"""
<ANNOTATED_PREMISES>
{newline.join([p.to_string() for p in self.annotated_premises])}
</ANNOTATED_PREMISES>
<CONCLUSION>
{self.conclusion.to_string()}
<CONCLUSION>
"""

    @staticmethod
    def from_string(text):
        return AnnotatedTranslation(
            annotated_premises=[AnnotatedSentence.from_string(b) for b in get_blocks_of(text, "ANNOTATED_PREMISES")],
            conclusion=Sentence.from_string(get_blocks_of(text, "CONCLUSION")[0])
        )


def get_blocks_of(text, tag):
    res = []
    for block in text.split(f"<{tag}>")[1:]:
        res.append(block.split(f"</{tag}>")[0])
    return res


def get_variable(text, tag):
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith(tag.lower()):
            return ":".join(line.split(":")[1:])
