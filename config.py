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
    'isteaq ulab info.xlsx',
    'movie_scripts_nadim_info_sheet.xlsx',
]

# If EXCEL_FILES is empty or not set, it will use EXCEL_FILE
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

