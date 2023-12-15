from tree_node.datatypes import *
from tree_node.utils import parse_xml_from_string


sentence = Sentence.from_xml(parse_xml_from_string("""
<SENTENCE>
    <PREMISE> When a person reads a book, that person gains knowledge. </PREMISE>
    <FOL> all x. all y. (Person(x) & Reads(x, y) & Book(y) -> Gains(x, Knowledge)) </FOL>
</SENTENCE>
"""))

print(sentence)

