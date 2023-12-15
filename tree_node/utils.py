from eval.tasks.utils import evaluate


def evaluate_premise_conclusion(premises, conclusion):
    try:
        evaluate(premises=premises, conclusion=conclusion)
    except Exception as e:
        print(f"Error in parsing and/or evaluating LLM output: {e}")
        return "Error"
