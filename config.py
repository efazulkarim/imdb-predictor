# ============================================================
# CONFIGURATION
# ============================================================
# IMDb Rating Predictor - Configuration Settings
# Modify these values to match your dataset structure

# File paths
# Single Excel file (for backward compatibility)
EXCEL_FILE = 'isteaq ulab info.xlsx'

# Multiple Excel files (use this if you have multiple files)
# Can be a single file path (string) or a list of file paths
# Examples:
#   EXCEL_FILES = ['file1.xlsx', 'file2.xlsx']  # List of files
#   EXCEL_FILES = 'single_file.xlsx'             # Single file as string
#   EXCEL_FILES = []                             # Empty list (will use EXCEL_FILE instead)
EXCEL_FILES = [
    'movie_lengths.xlsx',
]

# If EXCEL_FILES is empty or not set, it will use EXCEL_FILE
SCRIPTS_DIR = 'scripts/'

# Excel column names (match your Excel file headers exactly)
SCRIPT_COL = '.txt Files'
RATING_COL = 'IMDb Rating'
YEAR_COL = 'Year'
DECADE_COL = 'Decade'
MOVIE_NAME_COL = 'Movie name'
MOVIE_LENGTH_COL = 'Movie length'

# Model training parameters
TEST_SIZE = 0.15          # 15% test (from total)
VALIDATION_SIZE = 0.15    # 15% validation (from total) 
RANDOM_STATE = 42

# ============================================================
# SBERT (Sentence Transformer) Configuration
# ============================================================
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast + good quality (384-dim output)
SBERT_EMBEDDING_DIM = 384              # Output dimension of all-MiniLM-L6-v2

# Chunking for long scripts (SBERT has ~256 token limit)
CHUNK_SIZE = 256          # Words per chunk
CHUNK_OVERLAP = 50        # Overlapping words between chunks for context continuity

# ============================================================
# Legacy Settings (kept for reference)
# ============================================================
# Word2Vec (deprecated - now using SBERT)
EMBEDDING_DIM = 100       # Word2Vec vector dimension
W2V_WINDOW = 5            # Context window size
W2V_MIN_COUNT = 3         # Minimum word frequency

# TF-IDF (deprecated)
MAX_TFIDF_FEATURES = 8000  # Vocabulary size for TF-IDF
