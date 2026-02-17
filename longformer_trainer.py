# ============================================================
# LONGFORMER + METADATA FUSION -- TRAINING SCRIPT
# ============================================================
# Before running in Google Colab:
#   Runtime  ->  Change runtime type  ->  Select GPU (T4)
#
# This script is the single entry-point for training.
# It loads data via the existing data_loader, scales metadata,
# builds DataLoaders, trains with mixed-precision & gradient
# accumulation, applies early stopping, and saves the best
# checkpoint to Google Drive.
# ============================================================

import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LongformerTokenizerFast, get_linear_schedule_with_warmup

from data_loader import load_dataset
from longformer_config import (
    BATCH_SIZE,
    DRIVE_SAVE_PATH,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FUSION_HIDDEN_DIM,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MAX_LENGTH,
    METADATA_EMBED_DIM,
    METADATA_INPUT_DIM,
    MODEL_NAME,
    RANDOM_STATE,
    WEIGHT_DECAY,
)
from longformer_dataset import MovieScriptDataset
from longformer_model import LongformerRegressor

warnings.filterwarnings("ignore")

# ============================================================
# GPU SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("  DEVICE INFORMATION")
print("=" * 60)
print(f"  Using device : {device}")
if torch.cuda.is_available():
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"  VRAM         : {gpu_mem:.1f} GB")
else:
    print("  WARNING: No GPU detected. Training will be extremely slow.")
    print("  Go to Runtime -> Change runtime type -> Select GPU")
print("=" * 60)


# ============================================================
# 1. LOAD & SPLIT DATA
# ============================================================
def load_and_split_data():
    """Load dataset via existing data_loader and perform 70/15/15 split.

    Returns
    -------
    Tuple of (texts, ratings, metadata) for train / val / test splits,
    plus the raw features_df for reference.
    """
    scripts_text, ratings, features_df, movie_names, script_files, decade_encoder = (
        load_dataset()
    )

    # Extract the 3 metadata columns
    metadata_cols = ["year", "decade_encoded", "movie_length"]
    metadata_raw = features_df[metadata_cols].copy()

    # Fill missing movie_length with median (will re-compute on train split later)
    median_length = metadata_raw["movie_length"].median()
    metadata_raw["movie_length"] = metadata_raw["movie_length"].fillna(median_length)
    metadata_np = metadata_raw.values.astype(np.float32)

    # Convert texts list to numpy array for easy indexing
    texts_arr = np.array(scripts_text, dtype=object)

    # First split: 70% train, 30% temp
    train_texts, temp_texts, train_ratings, temp_ratings, train_meta, temp_meta = (
        train_test_split(
            texts_arr,
            ratings,
            metadata_np,
            test_size=0.30,
            random_state=RANDOM_STATE,
        )
    )

    # Second split: 50/50 on the 30% temp -> 15% val, 15% test
    val_texts, test_texts, val_ratings, test_ratings, val_meta, test_meta = (
        train_test_split(
            temp_texts,
            temp_ratings,
            temp_meta,
            test_size=0.50,
            random_state=RANDOM_STATE,
        )
    )

    print(f"\n>> Data Split:")
    print(f"   Train      : {len(train_texts)} samples")
    print(f"   Validation : {len(val_texts)} samples")
    print(f"   Test       : {len(test_texts)} samples")

    return (
        train_texts.tolist(), train_ratings, train_meta,
        val_texts.tolist(), val_ratings, val_meta,
        test_texts.tolist(), test_ratings, test_meta,
    )


# ============================================================
# 2. SCALE METADATA (fit on train only)
# ============================================================
def scale_metadata(train_meta, val_meta, test_meta):
    """Fit StandardScaler on training metadata, transform all splits.

    Returns scaled arrays and the fitted scaler (saved with checkpoint).
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_meta)
    val_scaled = scaler.transform(val_meta)
    test_scaled = scaler.transform(test_meta)

    print("\n>> Metadata scaling (StandardScaler):")
    print(f"   Features : year, decade_encoded, movie_length")
    print(f"   Train mean: {scaler.mean_}")
    print(f"   Train std : {scaler.scale_}")

    return train_scaled, val_scaled, test_scaled, scaler


# ============================================================
# 3. CREATE DATALOADERS
# ============================================================
def create_dataloaders(
    train_texts, train_ratings, train_meta_scaled,
    val_texts, val_ratings, val_meta_scaled,
    test_texts, test_ratings, test_meta_scaled,
):
    """Instantiate tokenizer, Datasets, and DataLoaders."""

    print(f"\n>> Loading tokenizer: {MODEL_NAME}")
    tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = MovieScriptDataset(
        train_texts, train_ratings, train_meta_scaled, tokenizer, MAX_LENGTH,
    )
    val_dataset = MovieScriptDataset(
        val_texts, val_ratings, val_meta_scaled, tokenizer, MAX_LENGTH,
    )
    test_dataset = MovieScriptDataset(
        test_texts, test_ratings, test_meta_scaled, tokenizer, MAX_LENGTH,
    )

    # num_workers=0 prevents multiprocessing issues on Colab
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    print(f"   Tokenizer loaded  (max_length={MAX_LENGTH})")
    print(f"   Train batches     : {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches      : {len(test_loader)}")

    return tokenizer, train_loader, val_loader, test_loader


# ============================================================
# 4. TRAIN ONE EPOCH
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, criterion):
    """Run one training epoch with mixed precision and gradient accumulation."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc="  Training", leave=False)
    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["rating"].to(device)

        with autocast():
            predictions = model(input_ids, attention_mask, metadata)
            loss = criterion(predictions, labels)
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM_STEPS  # un-scale for logging
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    torch.cuda.empty_cache()
    return total_loss / max(num_batches, 1)


# ============================================================
# 5. EVALUATE
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion):
    """Evaluate model and return loss + regression metrics."""

    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    progress = tqdm(dataloader, desc="  Evaluating", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["rating"].to(device)

        with autocast():
            predictions = model(input_ids, attention_mask, metadata)
            loss = criterion(predictions, labels)

        total_loss += loss.item()
        num_batches += 1
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)

    return avg_loss, mse, rmse, r2


# ============================================================
# 6. MAIN TRAINING LOOP
# ============================================================
def main():
    # ---- Mount Google Drive ----
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("\n>> Google Drive mounted successfully.")
    except ImportError:
        print("\n>> Not running on Colab -- skipping Drive mount.")
        print("   Model will be saved locally instead.")

    # ---- Load data ----
    (
        train_texts, train_ratings, train_meta,
        val_texts, val_ratings, val_meta,
        test_texts, test_ratings, test_meta,
    ) = load_and_split_data()

    # ---- Scale metadata ----
    train_meta_s, val_meta_s, test_meta_s, metadata_scaler = scale_metadata(
        train_meta, val_meta, test_meta,
    )

    # ---- Create DataLoaders ----
    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(
        train_texts, train_ratings, train_meta_s,
        val_texts, val_ratings, val_meta_s,
        test_texts, test_ratings, test_meta_s,
    )

    # ---- Build model ----
    print(f"\n>> Building LongformerRegressor")
    print(f"   Pretrained : {MODEL_NAME}")
    print(f"   Dropout    : {DROPOUT}")
    print(f"   Meta dims  : {METADATA_INPUT_DIM} -> {METADATA_EMBED_DIM}")
    print(f"   Fusion     : {768 + METADATA_EMBED_DIM} -> {FUSION_HIDDEN_DIM} -> 1")

    model = LongformerRegressor()
    model.to(device)

    # ---- Optimizer, scheduler, criterion, scaler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    total_training_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = total_training_steps // 10  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    print(f"\n>> Training configuration:")
    print(f"   Epochs              : {EPOCHS}")
    print(f"   Batch size          : {BATCH_SIZE}")
    print(f"   Grad accum steps    : {GRAD_ACCUM_STEPS}")
    print(f"   Effective batch     : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   Learning rate       : {LEARNING_RATE}")
    print(f"   Weight decay        : {WEIGHT_DECAY}")
    print(f"   Total steps         : {total_training_steps}")
    print(f"   Warmup steps        : {warmup_steps}")
    print(f"   Early stopping      : {EARLY_STOPPING_PATIENCE} epochs patience")
    print(f"   Mixed precision     : enabled")

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
        )

        # Validate
        val_loss, val_mse, val_rmse, val_r2 = evaluate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\n  Epoch {epoch}/{EPOCHS}  ({epoch_time:.0f}s)")
        print(f"    Train Loss : {train_loss:.4f}")
        print(f"    Val Loss   : {val_loss:.4f}")
        print(f"    Val RMSE   : {val_rmse:.4f}")
        print(f"    Val R²     : {val_r2:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_rmse": val_rmse,
                "best_val_r2": val_r2,
                "epoch": epoch,
                "metadata_scaler_mean": metadata_scaler.mean_.tolist(),
                "metadata_scaler_scale": metadata_scaler.scale_.tolist(),
                "config": {
                    "model_name": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                    "dropout": DROPOUT,
                    "metadata_input_dim": METADATA_INPUT_DIM,
                    "metadata_embed_dim": METADATA_EMBED_DIM,
                    "fusion_hidden_dim": FUSION_HIDDEN_DIM,
                },
            }

            save_path = DRIVE_SAVE_PATH
            try:
                torch.save(checkpoint, save_path)
                print(f"    >> Best model saved to {save_path}")
            except OSError:
                # Fallback to local save if Drive is not available
                local_path = "longformer_imdb_model.pt"
                torch.save(checkpoint, local_path)
                print(f"    >> Best model saved locally to {local_path}")

            # Save tokenizer alongside checkpoint
            try:
                tokenizer_dir = save_path.replace(".pt", "_tokenizer")
                tokenizer.save_pretrained(tokenizer_dir)
            except OSError:
                tokenizer.save_pretrained("longformer_imdb_tokenizer")

        else:
            patience_counter += 1
            print(f"    >> No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

        torch.cuda.empty_cache()

    # ---- Final evaluation on test set ----
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best checkpoint
    try:
        checkpoint = torch.load(DRIVE_SAVE_PATH, map_location=device)
    except OSError:
        checkpoint = torch.load("longformer_imdb_model.pt", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_loss, test_mse, test_rmse, test_r2 = evaluate(model, test_loader, criterion)

    print(f"\n  Test MSE  : {test_mse:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}")
    print(f"  Test R²   : {test_r2:.4f}")
    print(f"\n  Best validation loss was {checkpoint['best_val_loss']:.4f} at epoch {checkpoint['epoch']}")
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
