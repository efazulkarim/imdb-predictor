# ============================================================
# CONFIGURATION
# ============================================================
# IMDb Rating Predictor - Configuration Settings
# Modify these values to match your dataset structure

# File paths
EXCEL_FILE = 'isteaq ulab info.xlsx'
SCRIPTS_DIR = 'scripts/'

# Excel column names (match your Excel file headers exactly)
SCRIPT_COL = '.txt Files'
RATING_COL = 'IMDb Rating'
YEAR_COL = 'Year'
DECADE_COL = 'Decade'
MOVIE_NAME_COL = 'Movie Name'

# Model training parameters
TEST_SIZE = 0.30          # 30% test, 70% train
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 8000  # Vocabulary size for TF-IDF

