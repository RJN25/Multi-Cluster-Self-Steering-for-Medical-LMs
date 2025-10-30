import os, torch
from config import SAVE_DIR, VEC_PATH, PROJ_PATH

os.makedirs(SAVE_DIR, exist_ok=True)

def save_vectors(vec_dict):  # {'A':tensor(d),...}
    torch.save({k:v.detach().cpu() for k,v in vec_dict.items()}, VEC_PATH)

def load_vectors():
    return torch.load(VEC_PATH, map_location="cpu")

def save_proj(module):
    torch.save(module.state_dict(), PROJ_PATH)
