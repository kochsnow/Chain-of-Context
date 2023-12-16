from eval.tasks.utils import evaluate
# from lxml import etree
# import xml.etree.ElementTree as ET
# import re


def evaluate_premise_conclusion(premises, conclusion):
    try:
        return evaluate(premises=premises, conclusion=conclusion)
    except Exception as e:
        print(f"Error in parsing and/or evaluating LLM output: {e}")
        return "Error"


# def parse_xml_from_string(text: str):
#     text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
#     text = re.sub(r'&lt;(\w+)&gt;', r'<\1>', text)
#     text = re.sub(r'&lt;/(\w+)&gt;', r'</\1>', text)
#     text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
#     tree = ET.fromstring(text)
#     return tree
