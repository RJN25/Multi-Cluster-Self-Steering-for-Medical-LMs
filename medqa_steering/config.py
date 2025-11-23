MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TARGET_LAYER = -1                  # last hidden layer
ALPHA = 0.3                        # steering strength (tune) test: 0, 0.3, 0.5, 0. 8
COSINE_GATE = None                 # only steer if cos(W·h, v*) < τ
MAX_NEW_TOKENS = 1                 # MCQ: next-token scoring only

SAVE_DIR = "artifacts"
VEC_PATH = f"{SAVE_DIR}/class_vectors.pt"          # dict{'A':vA,...}
PROJ_PATH = f"{SAVE_DIR}/proj_W.pt"                # torch Linear (d→d)

LOG_DIR = "logs"
SEED = 42
DEVICE = "cuda"
BATCH_SIZE = 8
NUM_WORKERS = 2
