from .medqa_dataset2 import LETTER

MCQ_TEMPLATE = """You are a medical assistant. Choose the single best answer.
Question:
{stem}

Options:
A) {a}
B) {b}
C) {c}
D) {d}

Reply ONLY with tags:
<analysis> ... </analysis>
<answer> A/B/C/D </answer>
"""

def build_prompt(stem, choices):
    a,b,c,d = choices
    return MCQ_TEMPLATE.format(stem=stem, a=a, b=b, c=c, d=d)
