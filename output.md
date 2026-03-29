======================================================================
🎬 IMDb RATING PREDICTOR - TRAINING PIPELINE
📁 Dataset: Movie Scripts + Excel Metadata
======================================================================
======================================================================
LOADING DATASET
======================================================================

> > Loading from single Excel file:
> > [OK] movie_lengths.xlsx: 5204 records (Ratings: 1.5-9.3)

[OK] No duplicates found - all 5204 records are unique

> > Combined dataset: 5204 unique records
> > Columns (8 total): Movie name, Year, IMDb Rating, IMDb ID, .txt Files, Collected By, Decade, Movie length
> > [OK] All expected columns present
> > Rating range: 1.5 - 9.3 (mean: 5.98)
> > Note: Scripts not found will be skipped automatically

> > Loading scripts from: scripts/
> > Processing.......... Done!

> > Loading Summary:
> > [OK] Successfully loaded: 5195 scripts
> > [X] Skipped: 9 total

      - Missing files: 0
      - Too short (<1KB): 9
      - Invalid rating: 0
      - Read errors: 0

======================================================================
MODEL TRAINING (70% Train / 15% Validation / 15% Test)
======================================================================

💾 Test set info saved: 'test_set_info.json'

📊 Data Split:
Training: 3636 samples (70%)
Validation: 779 samples (15%)
Testing: 780 samples (15%)
Rating range: 1.5 - 9.3
Mean rating: 5.98 (std: 1.44)

⚖️ Sample Weights (Class Balancing):
Low (1-4): n= 371, weight=5.25x
Medium (4-6): n=1947, weight=1.00x
Good (6-8): n=1171, weight=1.66x
Excellent (8-10): n= 147, weight=13.24x

🔤 Loading SBERT model: 'all-MiniLM-L6-v2'...
Model loaded. Embedding dimension: 384
Chunk size: 256 words, Overlap: 50 words

📝 Generating SBERT embeddings for training set...
Embedding progress: 500/3636 scripts...
Embedding progress: 1000/3636 scripts...
Embedding progress: 1500/3636 scripts...
Embedding progress: 2000/3636 scripts...
Embedding progress: 2500/3636 scripts...
Embedding progress: 3000/3636 scripts...
Embedding progress: 3500/3636 scripts...
Training embeddings shape: (3636, 384)

📝 Generating SBERT embeddings for validation set...

📝 Generating SBERT embeddings for test set...
Numerical features: 19
Combined features: 403 (SBERT: 384 + Numerical: 19)

---

## MODEL EVALUATION RESULTS

🔄 Training Random Forest...
Validation: RMSE=1.0214 | MAE=0.7677 | R²=0.4895
Test: RMSE=1.0959 | MAE=0.8260 | R²=0.4787

🔄 Training Gradient Boosting...
Validation: RMSE=0.9646 | MAE=0.7239 | R²=0.5447
Test: RMSE=1.0039 | MAE=0.7602 | R²=0.5626

🔄 Training LightGBM...
Validation: RMSE=0.9887 | MAE=0.7469 | R²=0.5217
Test: RMSE=0.9969 | MAE=0.7616 | R²=0.5686

🔄 Training XGBoost...
Validation: RMSE=0.9419 | MAE=0.7067 | R²=0.5659
Test: RMSE=0.9881 | MAE=0.7509 | R²=0.5762

🔄 Training Ridge Regression...
Validation: RMSE=0.9754 | MAE=0.7486 | R²=0.5344
Test: RMSE=1.0356 | MAE=0.8039 | R²=0.5345

🔄 Training ElasticNet...
Validation: RMSE=1.0842 | MAE=0.8324 | R²=0.4248
Test: RMSE=1.1418 | MAE=0.8880 | R²=0.4342

======================================================================
🏆 BEST MODEL: XGBoost
======================================================================
Validation RMSE: 0.9419
Validation MAE: 0.7067
Validation R²: 0.5659

---

Test RMSE: 0.9881
Test MAE: 0.7509
Test R²: 0.5762

======================================================================
PREDICTION ANALYSIS (Test Set)
======================================================================

## 📊 Error Distribution:

Within ±0.5: 345 ( 44.2%) ██████████████████████
Within ±1.0: 559 ( 71.7%) ███████████████████████████████████
Within ±1.5: 673 ( 86.3%) ███████████████████████████████████████████
Within ±2.0: 741 ( 95.0%) ███████████████████████████████████████████████

## 📋 Sample Predictions (15 random):

     Actual | Predicted | Error

---

       3.60 | 4.24 | +0.64
       5.00 | 5.60 | +0.60
       5.80 | 5.87 | +0.07
       4.10 | 5.75 | +1.65
       5.30 | 4.67 | -0.63
       5.90 | 5.00 | -0.90
       7.70 | 7.73 | +0.03
       7.40 | 7.72 | +0.32
       5.00 | 4.90 | -0.10
       5.40 | 5.42 | +0.02
       5.60 | 5.12 | -0.48
       7.70 | 7.49 | -0.21
       5.80 | 5.21 | -0.59
       7.30 | 7.34 | +0.04
       5.40 | 5.70 | +0.30

## 📈 Performance by Rating Range:

Low (1-4) : MAE = 1.541 (n=107)
Medium (4-6) : MAE = 0.579 (n=380)
Good (6-8) : MAE = 0.659 (n=250)
Excellent (8-10) : MAE = 0.839 (n=43)

💾 Model saved: 'imdb_model.pkl' (1.2 MB)
Note: SBERT model 'all-MiniLM-L6-v2' will be loaded at inference time.

======================================================================
📖 HOW TO USE THE TRAINED MODEL
======================================================================

    # In Python:
    from predictor import predict_rating; print(f'Predicted Rating: {predict_rating("scripts/file10074.txt")}/10')

    # Predict from text
    script = "JOHN: Hello!\nMARY: Hi there!"
    rating = predict_from_text(script)

✅ Training complete!
