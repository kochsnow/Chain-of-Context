from .TreeNode import TreeNode
from eval.base import OWAFOLTask
from tree_node.datatypes import Translation, Sentence, RawPremiseAndConclusion, AnnotatedTranslation
from .utils import evaluate_premise_conclusion


class ResultTreeNode(TreeNode):
    def __init__(self, doc, annotated_translation: AnnotatedTranslation, task_name, task: OWAFOLTask, model, chat):
        super(ResultTreeNode, self).__init__(doc, model, task_name, task)

        self.annotated_translation = annotated_translation
        premises = []
        for annot in self.annotated_translation.annotated_premises:
            premises.append(annot.sentence.fol)
            for context in annot.contexts:
                premises.append(context.fol)
        self.result = evaluate_premise_conclusion(
            premises=premises,
            conclusion=self.annotated_translation.conclusion.fol
        )

    def name(self):
        return f"{self.model}_{self.task_name}.json"

    def expand(self):
        return []

    def get_children_votes(self):
        votes = {"True": 0, "False": 0, "Uncertain": 0, "Error": 0}
        votes[self.result] += 1
        return votes


