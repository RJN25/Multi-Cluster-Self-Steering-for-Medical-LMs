from .medqa_dataset import LETTER

MCQ_TEMPLATE = """You are a medical assistant. Choose the single best answer.
Question:
{stem}

Options:
A) {a}
B) {b}
C) {c}
D) {d}

Reply ONLY with tags:
<answer> A/B/C/D </answer>
<analysis> ... </analysis>
<confidence> 0.0–1.0 </confidence>
"""

def build_prompt(stem, choices):
    a,b,c,d = choices
    return MCQ_TEMPLATE.format(stem=stem, a=a, b=b, c=c, d=d)
