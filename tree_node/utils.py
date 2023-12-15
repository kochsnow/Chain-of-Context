from eval.tasks.utils import evaluate
from lxml import etree
import xml.etree.ElementTree as ET


def evaluate_premise_conclusion(premises, conclusion):
    try:
        evaluate(premises=premises, conclusion=conclusion)
    except Exception as e:
        print(f"Error in parsing and/or evaluating LLM output: {e}")
        return "Error"


def parse_xml_from_string(text: str):
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(text, parser=parser)
    return tree
