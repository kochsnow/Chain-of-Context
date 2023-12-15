from dataclasses import dataclass
from typing import List


@dataclass
class Sentence:
    premise: str
    fol: str

    def format(self):
        return f"""
<SENTENCE>
    PREMISE: {self.premise}
    FOL: {self.fol}
</SENTENCE>
"""


@dataclass
class Translation:
    premises: List[Sentence]
    conclusion: Sentence

    # def format(self):
    #     ans = ""
    #     for sentence in self.premises:
    #         return



def to_sentence(line0, line1):
    line0 = line0.strip()
    line1 = line1.strip()

    premise = None
    fol = None

    if line0.lower().startswith("fol:"):
        line0 = line0[len("fol:"):]
        fol = line0
    if line1.lower().startswith("sentence:"):
        line1 = line1[len("sentence:"):]
        premise = line1
    if line1.lower().startswith("text:"):
        line1 = line1[len("text:"):]
        premise = line1
    if line1.lower().startswith("premise:"):
        line1 = line1[len("premise:"):]
        premise = line1
    return Sentence(premise=premise, fol=fol)
