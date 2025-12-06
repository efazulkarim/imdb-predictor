 IMDb RATING PREDICTOR - OPTIMIZED FOR MOVIE SCRIPTS
# Version: 2.0 (Production Ready)
# Dataset: 5000 Movie Scripts + Excel Metadata
# Split: 70% Training / 30% Testing
# ============================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
EXCEL_FILE = 'isteaq ulab info.xlsx'
SCRIPTS_DIR = 'scripts/'
SCRIPT_COL = '.txt Files'
RATING_COL = 'IMDb Rating'
YEAR_COL = 'Year'
DECADE_COL = 'Decade'
MOVIE_NAME_COL = 'Movie Name'

TEST_SIZE = 0.30          # 30% test, 70% train
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 8000  # Vocabulary size for TF-IDF


# ============================================================
# TEXT PREPROCESSING
# ============================================================
class ScriptPreprocessor:
    """Advanced preprocessing for movie scripts."""

    @staticmethod
    def clean_text(text):
        """Clean and normalize script text."""
        # Convert to lowercase
        text = text.lower()

        # Remove stage directions [GUNSHOT], (CRYING), etc.
        text = re.sub(r'\[.*?\]', ' ', text)
        text = re.sub(r'\(.*?\)', ' ', text)

        # Remove character names (JOHN:, MARY:, etc.)
        text = re.sub(r'^[A-Z][A-Z\s]+:', ' ', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove timestamps and scene numbers
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', ' ', text)
        text = re.sub(r'\bscene\s*\d+\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bint\.?\b|\bext\.?\b', ' ', text, flags=re.IGNORECASE)

        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9'.,!?\s]", ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def extract_features(raw_text):
        """Extract rich numerical features from script."""
        features = {}

        # === Basic Statistics ===
        features['char_count'] = len(raw_text)
        words = raw_text.split()
        features['word_count'] = len(words)
        lines = raw_text.split('\n')
        features['line_count'] = len(lines)

        # === Vocabulary Complexity ===
        if words:
            features['avg_word_length'] = np.mean([len(w) for w in words])
            features['unique_word_ratio'] = len(set(words)) / len(words)
            # Long words (8+ chars) ratio - indicates sophisticated vocabulary
            long_words = [w for w in words if len(w) >= 8]
            features['long_word_ratio'] = len(long_words) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_word_ratio'] = 0
            features['long_word_ratio'] = 0

        # === Sentence Structure ===
        sentences = re.split(r'[.!?]+', raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features['sentence_count'] = len(sentences)
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = np.mean(sent_lengths)
            features['sentence_length_std'] = np.std(sent_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_std'] = 0

        # === Dialogue Analysis ===
        # Character name patterns (ALL CAPS followed by colon)
        character_lines = re.findall(r'^[A-Z][A-Z\s]+:', raw_text, re.MULTILINE)
        features['dialogue_density'] = len(character_lines) / max(len(lines), 1)

        # Unique characters speaking
        unique_chars = set([c.strip(':').strip() for c in character_lines])
        features['unique_characters'] = len(unique_chars)

        # === Emotional Indicators ===
        features['exclamation_ratio'] = raw_text.count('!') / max(features['word_count'], 1) * 100
        features['question_ratio'] = raw_text.count('?') / max(features['word_count'], 1) * 100

        # === Action/Direction Density ===
        action_brackets = len(re.findall(r'\[.*?\]', raw_text))
        action_parens = len(re.findall(r'\(.*?\)', raw_text))
        features['action_density'] = (action_brackets + action_parens) / max(features['word_count'], 1) * 100

        # === Script Structure Indicators ===
        # Scene headings (INT., EXT.)
        scene_headings = len(re.findall(r'\b(INT|EXT)\.?\s', raw_text, re.IGNORECASE))
        features['scene_count'] = scene_headings

        # Pacing: words per scene (higher = slower pacing)
        features['words_per_scene'] = features['word_count'] / max(scene_headings, 1)

        return features


# ============================================================
# DATA LOADING
# ============================================================
def load_dataset():
    """Load Excel and script files, extract all features."""
    print("=" * 70)
    print("  LOADING DATASET")
    print("=" * 70)

    # Load Excel
    df = pd.read_excel(EXCEL_FILE)
    print(f"\n📊 Excel loaded: {len(df)} records")
    print(f"   Columns: {list(df.columns)}")

    # Data containers
    scripts_text = []
    script_features = []
    ratings = []
    years = []
    decades = []
    movie_names = []

    # Track loading stats
    loaded = 0
    skipped = {'missing': 0, 'short': 0, 'invalid_rating': 0, 'error': 0}

    print(f"\n📁 Loading scripts from: {SCRIPTS_DIR}")
    print("   Processing", end="", flush=True)

    for idx, row in df.iterrows():
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(".", end="", flush=True)

        # Skip if rating is missing or invalid
        try:
            rating = float(row[RATING_COL])
            if rating < 0 or rating > 10 or pd.isna(rating):
                skipped['invalid_rating'] += 1
                continue
        except:
            skipped['invalid_rating'] += 1
            continue

        # Get script filename
        script_file = str(row[SCRIPT_COL]).strip()

        # Try multiple filename patterns
        possible_paths = [
            os.path.join(SCRIPTS_DIR, script_file),
            os.path.join(SCRIPTS_DIR, script_file + '.txt'),
            os.path.join(SCRIPTS_DIR, script_file.replace(' ', '-') + '.txt'),
            os.path.join(SCRIPTS_DIR, script_file.replace(' ', '-')),
            os.path.join(SCRIPTS_DIR, script_file.replace(' ', '_') + '.txt'),
        ]

        # Extract number and try more patterns
        num_match = re.search(r'(\d+)', script_file)
        if num_match:
            num = num_match.group(1)
            possible_paths.extend([
                os.path.join(SCRIPTS_DIR, f'file-{num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file_{num}.txt'),
                os.path.join(SCRIPTS_DIR, f'file {num}.txt'),
            ])

        # Find existing file
        filepath = None
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break

        if not filepath:
            skipped['missing'] += 1
            continue

        # Read script
        try:
            raw_text = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=enc) as f:
                        raw_text = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if not raw_text or len(raw_text) < 1000:
                skipped['short'] += 1
                continue

            # Preprocess and extract features
            cleaned_text = ScriptPreprocessor.clean_text(raw_text)
            features = ScriptPreprocessor.extract_features(raw_text)

            # Store data
            scripts_text.append(cleaned_text)
            script_features.append(features)
            ratings.append(rating)
            years.append(int(row[YEAR_COL]) if pd.notna(row.get(YEAR_COL)) else 2000)
            decades.append(str(row.get(DECADE_COL, '2000s')))
            movie_names.append(str(row.get(MOVIE_NAME_COL, f'Movie_{idx}')))
            loaded += 1

        except Exception as e:
            skipped['error'] += 1
            continue

    print(" Done!")

    # Summary
    print(f"\n📈 Loading Summary:")
    print(f"   ✅ Successfully loaded: {loaded} scripts")
    print(f"   ❌ Skipped: {sum(skipped.values())} total")
    print(f"      - Missing files: {skipped['missing']}")
    print(f"      - Too short (<1KB): {skipped['short']}")
    print(f"      - Invalid rating: {skipped['invalid_rating']}")
    print(f"      - Read errors: {skipped['error']}")

    # Convert to arrays/dataframes
    features_df = pd.DataFrame(script_features)
    features_df['year'] = years

    # Encode decades
    le = LabelEncoder()
    features_df['decade_encoded'] = le.fit_transform(decades)

    return scripts_text, np.array(ratings), features_df, movie_names, le


# ============================================================
# MODEL TRAINING
# ============================================================
def train_and_evaluate(scripts_text, ratings, features_df):
    """Train multiple models, evaluate, and return the best."""
    print("\n" + "=" * 70)
    print("  MODEL TRAINING (70% Train / 30% Test)")
    print("=" * 70)

    # === Data Split ===
    (X_text_train, X_text_test, 
     y_train, y_test,
     X_feat_train, X_feat_test) = train_test_split(
        scripts_text, ratings, features_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_text_train)} samples ({100-TEST_SIZE*100:.0f}%)")
    print(f"   Testing:  {len(X_text_test)} samples ({TEST_SIZE*100:.0f}%)")
    print(f"   Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
    print(f"   Mean rating: {ratings.mean():.2f} (std: {ratings.std():.2f})")

    # === TF-IDF Vectorization ===
    print(f"\n🔤 Creating TF-IDF features (max {MAX_TFIDF_FEATURES} features)...")

    tfidf = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),       # Unigrams + Bigrams
        min_df=3,                  # Minimum document frequency
        max_df=0.85,               # Maximum document frequency
        sublinear_tf=True,         # Apply log scaling
        dtype=np.float32
    )

    X_tfidf_train = tfidf.fit_transform(X_text_train)
    X_tfidf_test = tfidf.transform(X_text_test)

    print(f"   TF-IDF shape: {X_tfidf_train.shape}")
    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")

    # === Scale Numerical Features ===
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_feat_train)
    X_num_test = scaler.transform(X_feat_test)

    print(f"   Numerical features: {X_num_train.shape[1]}")

    # === Combine Features ===
    X_train = hstack([X_tfidf_train, csr_matrix(X_num_train)])
    X_test = hstack([X_tfidf_test, csr_matrix(X_num_test)])

    print(f"   Combined features: {X_train.shape[1]}")

    # === Define Models ===
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_split=10,
            random_state=RANDOM_STATE
        ),
        'Ridge Regression': Ridge(
            alpha=1.5,
            random_state=RANDOM_STATE
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=RANDOM_STATE
        )
    }

    # === Train and Evaluate ===
    print("\n" + "-" * 70)
    print("  MODEL EVALUATION RESULTS")
    print("-" * 70)

    results = {}
    best_model_name = None
    best_rmse = float('inf')

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 1.0, 10.0)  # Clip to valid IMDb range

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'predictions': y_pred,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        print(f"   RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  R²: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

    # === Best Model Summary ===
    print("\n" + "=" * 70)
    print(f"  🏆 BEST MODEL: {best_model_name}")
    print("=" * 70)
    print(f"   RMSE: {results[best_model_name]['RMSE']:.4f} (average prediction error)")
    print(f"   MAE:  {results[best_model_name]['MAE']:.4f}")
    print(f"   R²:   {results[best_model_name]['R2']:.4f}")

    return results, best_model_name, tfidf, scaler, y_test


# ============================================================
# DETAILED ANALYSIS
# ============================================================
def detailed_analysis(results, best_model_name, y_test):
    """Provide detailed prediction analysis."""
    print("\n" + "=" * 70)
    print("  PREDICTION ANALYSIS")
    print("=" * 70)

    y_pred = results[best_model_name]['predictions']

    # Error analysis
    errors = np.abs(y_pred - y_test)

    print("\n📊 Error Distribution:")
    print("-" * 40)
    brackets = [(0.5, '±0.5'), (1.0, '±1.0'), (1.5, '±1.5'), (2.0, '±2.0')]
    for threshold, label in brackets:
        count = np.sum(errors <= threshold)
        pct = count / len(errors) * 100
        bar = '█' * int(pct / 2)
        print(f"   Within {label}: {count:4d} ({pct:5.1f}%) {bar}")

    # Sample predictions
    print("\n📋 Sample Predictions (15 random):")
    print("-" * 50)
    print(f"   {'Actual':>8} | {'Predicted':>10} | {'Error':>8}")
    print("-" * 50)

    indices = np.random.choice(len(y_test), min(15, len(y_test)), replace=False)
    for i in indices:
        actual = y_test[i]
        pred = y_pred[i]
        error = pred - actual
        print(f"   {actual:8.2f} | {pred:10.2f} | {error:+8.2f}")

    # Rating range performance
    print("\n📈 Performance by Rating Range:")
    print("-" * 50)
    ranges = [(1, 4, 'Low (1-4)'), (4, 6, 'Medium (4-6)'), 
              (6, 8, 'Good (6-8)'), (8, 10, 'Excellent (8-10)')]

    for low, high, label in ranges:
        mask = (y_test >= low) & (y_test < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(np.abs(y_pred[mask] - y_test[mask]))
            count = np.sum(mask)
            print(f"   {label:20s}: MAE = {range_mae:.3f} (n={count})")


# ============================================================
# SAVE MODEL
# ============================================================
def save_model(results, best_model_name, tfidf, scaler, decade_encoder):
    """Save trained model and preprocessors."""
    model_package = {
        'model': results[best_model_name]['model'],
        'tfidf': tfidf,
        'scaler': scaler,
        'decade_encoder': decade_encoder,
        'model_name': best_model_name,
        'metrics': {
            'RMSE': results[best_model_name]['RMSE'],
            'MAE': results[best_model_name]['MAE'],
            'R2': results[best_model_name]['R2']
        }
    }

    with open('imdb_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)

    size_mb = os.path.getsize('imdb_model.pkl') / (1024 * 1024)
    print(f"\n💾 Model saved: 'imdb_model.pkl' ({size_mb:.1f} MB)")


# ============================================================
# PREDICTION FUNCTIONS (Use after training)
# ============================================================
def predict_rating(script_path, model_path='imdb_model.pkl'):
    """
    Predict IMDb rating for a new script file.

    Usage:
        rating = predict_rating('path/to/script.txt')
        print(f"Predicted Rating: {rating}/10")
    """
    # Load model package
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)

    # Read and process script
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    cleaned_text = ScriptPreprocessor.clean_text(raw_text)
    features = ScriptPreprocessor.extract_features(raw_text)

    # Add default year/decade
    features['year'] = 2020
    features['decade_encoded'] = 0

    # Transform
    X_tfidf = pkg['tfidf'].transform([cleaned_text])
    X_num = pkg['scaler'].transform(pd.DataFrame([features]))
    X = hstack([X_tfidf, csr_matrix(X_num)])

    # Predict
    rating = pkg['model'].predict(X)[0]
    return round(np.clip(rating, 1.0, 10.0), 2)


def predict_from_text(script_text, model_path='imdb_model.pkl'):
    """
    Predict IMDb rating from raw script text.

    Usage:
        script = "JOHN: Hello!\nMARY: Hi there..."
        rating = predict_from_text(script)
    """
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)

    cleaned_text = ScriptPreprocessor.clean_text(script_text)
    features = ScriptPreprocessor.extract_features(script_text)
    features['year'] = 2020
    features['decade_encoded'] = 0

    X_tfidf = pkg['tfidf'].transform([cleaned_text])
    X_num = pkg['scaler'].transform(pd.DataFrame([features]))
    X = hstack([X_tfidf, csr_matrix(X_num)])

    rating = pkg['model'].predict(X)[0]
    return round(np.clip(rating, 1.0, 10.0), 2)


def batch_predict(script_paths, model_path='imdb_model.pkl'):
    """
    Predict ratings for multiple scripts.

    Usage:
        paths = ['script1.txt', 'script2.txt', 'script3.txt']
        ratings = batch_predict(paths)
    """
    return [predict_rating(p, model_path) for p in script_paths]


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  🎬 IMDb RATING PREDICTOR - TRAINING PIPELINE")
    print("  📁 Dataset: Movie Scripts + Excel Metadata")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(EXCEL_FILE):
        print(f"\n❌ ERROR: Excel file not found: '{EXCEL_FILE}'")
        return

    if not os.path.exists(SCRIPTS_DIR):
        print(f"\n❌ ERROR: Scripts folder not found: '{SCRIPTS_DIR}'")
        return

    # Load data
    scripts_text, ratings, features_df, movie_names, decade_encoder = load_dataset()

    if len(scripts_text) < 100:
        print("\n⚠️  WARNING: Less than 100 scripts loaded!")
        print("   Results may be unreliable. Check file paths.")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            return

    # Train models
    results, best_model_name, tfidf, scaler, y_test = train_and_evaluate(
        scripts_text, ratings, features_df
    )

    # Detailed analysis
    detailed_analysis(results, best_model_name, y_test)

    # Save model
    save_model(results, best_model_name, tfidf, scaler, decade_encoder)

    # Usage instructions
    print("\n" + "=" * 70)
    print("  📖 HOW TO USE THE TRAINED MODEL")
    print("=" * 70)
    print("""
    # In Python:
    from imdb_predictor import predict_rating, predict_from_text

    # Predict from file
    rating = predict_rating('new_script.txt')
    print(f"Predicted: {rating}/10")

    # Predict from text
    script = "JOHN: Hello!\nMARY: Hi there!"
    rating = predict_from_text(script)
    """)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()