# Using Multiple Excel Files

This guide explains how to use multiple Excel files with the IMDb Rating Predictor.

## 📋 Overview

You can now load data from multiple Excel files that:
- Have the same column structure
- Share scripts in the `scripts/` folder
- May have overlapping entries (duplicates are automatically handled)

## 🔧 Configuration

### Option 1: Single Excel File (Default)

In `config.py`, use a single file:

```python
EXCEL_FILE = 'isteaq ulab info.xlsx'
EXCEL_FILES = []  # Leave empty or don't define
```

### Option 2: Multiple Excel Files

In `config.py`, specify multiple files:

```python
EXCEL_FILES = [
    'isteaq ulab info.xlsx',
    'second_file.xlsx',
    'third_file.xlsx',
]
```

**Note:** If `EXCEL_FILES` is defined and has entries, it will be used. Otherwise, it falls back to `EXCEL_FILE`.

## 📊 How It Works

1. **Loading**: All specified Excel files are loaded
2. **Combining**: Data from all files is combined into one dataset
3. **Deduplication**: If the same script filename appears in multiple files, only the first occurrence is kept
4. **Processing**: All unique scripts are processed normally

## ✅ Example Setup

### Step 1: Update `config.py`

```python
EXCEL_FILES = [
    'isteaq ulab info.xlsx',      # Your first file
    'additional_scripts.xlsx',    # Your second file
]
```

### Step 2: Ensure Both Files Have Same Columns

Both Excel files must have these columns (matching your config):
- `.txt Files` (or whatever `SCRIPT_COL` is set to)
- `IMDb Rating` (or whatever `RATING_COL` is set to)
- `Year` (optional)
- `Decade` (optional)
- `Movie Name` (optional)

### Step 3: Place Script Files

All `.txt` script files should be in the `scripts/` folder, regardless of which Excel file references them.

### Step 4: Run Training

```bash
python imdb_rating_predictor.py
```

The script will:
- Load both Excel files
- Show how many records from each file
- Combine them
- Remove duplicates (if any)
- Process all unique scripts

## 📈 Output Example

When using multiple files, you'll see output like:

```
======================================================================
  LOADING DATASET
======================================================================

📊 Loading from 2 Excel file(s):
   ✅ isteaq ulab info.xlsx: 2500 records
   ✅ additional_scripts.xlsx: 2000 records
   📝 Removed 150 duplicate script entries

📊 Combined dataset: 4350 unique records
   Columns: ['.txt Files', 'IMDb Rating', 'Year', ...]
```

## ⚠️ Important Notes

1. **Duplicate Handling**: If the same script filename appears in multiple files, only the first occurrence (from the first file in the list) is kept.

2. **Column Consistency**: All files must have the same column names. The script will use the columns defined in `config.py`.

3. **Script Files**: All script `.txt` files must be in the `scripts/` folder. The Excel files just reference them.

4. **Missing Files**: If a file in `EXCEL_FILES` doesn't exist, it will be skipped with a warning, but processing will continue with the available files.

## 🔍 Troubleshooting

### "No Excel files found"
- Check that file paths in `EXCEL_FILES` are correct
- Ensure files are in the same directory as the script, or use full paths

### "Columns don't match"
- Verify all Excel files have the same column headers
- Check `config.py` column names match your Excel files

### "Scripts not loading"
- Ensure all referenced script files exist in `scripts/` folder
- Check filename matching (the script tries multiple patterns)

## 💡 Tips

1. **File Organization**: Keep related scripts together in the same Excel file when possible
2. **Naming**: Use descriptive Excel file names to track data sources
3. **Validation**: Check the loading summary to see how many records came from each file
4. **Testing**: Start with one file, then add more to verify everything works





