import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, DEVICE

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        output_hidden_states=True,
        attn_implementation="eager"
    ).to(DEVICE).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model
