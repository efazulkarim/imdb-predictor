# ============================================================
# LONGFORMER DEEP LEARNING CONFIGURATION
# ============================================================
# Hyperparameters for the Longformer + Metadata Fusion model.
# Optimized for Google Colab free GPU (T4 / P100).
# ============================================================

# --- Pretrained model ---
MODEL_NAME = "allenai/longformer-base-4096"

# --- Tokenization ---
# 2048 is the safe default for free Colab (~15 GB VRAM).
# Switch to 4096 only if you have Colab Pro / A100.
MAX_LENGTH = 2048

# --- Training ---
BATCH_SIZE = 1               # single sample per GPU step
GRAD_ACCUM_STEPS = 4         # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 5
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 3
RANDOM_STATE = 42

# --- Metadata fusion dimensions ---
# Metadata features: year, decade_encoded, movie_length
METADATA_INPUT_DIM = 3
METADATA_EMBED_DIM = 32      # output size of the metadata FC branch
FUSION_HIDDEN_DIM = 128       # hidden layer after text + metadata concatenation

# --- Checkpoint path (Google Drive) ---
DRIVE_SAVE_PATH = "/content/drive/MyDrive/longformer_imdb_model.pt"
