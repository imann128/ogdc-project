"""
ogdc_loader_cleaner.py
----------------------
Loads and cleans OGDC (Oil & Gas Development Co.) stock price data
exported from Investing.com.

Usage:
    python ogdc_loader_cleaner.py

Output:
    - Diagnostic prints at each stage
    - ogdc_cleaned.csv saved to the working directory
    - Returns the cleaned DataFrame (importable as a module too)
"""
import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import processed, img, frontend, PATHS, ensure_dirs


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__)) # Scripts
# Data Directory
DATA_DIR = os.path.join(BASE, "..", "data", "raw")  
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "Oil and Gas Development Co Stock Price History.csv")
# Output Directory
OUTPUT_DIR = os.path.join(BASE, "..", "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUTPUT_DIR, "ogdc_cleaned.csv")
# Image Directory
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")
os.makedirs(IMG_DIR, exist_ok=True)
IMG_PRE = os.path.join(IMG_DIR, "trend_")

import pandas as pd
import sys
import warnings

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH      = DATA_PATH      # Input raw CSV
OUTPUT_PATH   = OUT_CSV        # Output cleaned CSV
EXPECTED_START = pd.Timestamp("2020-01-01")
EXPECTED_END   = pd.Timestamp("2026-04-01")

PRICE_COLS   = ["Price", "Open", "High", "Low"]
VOLUME_COL   = "Vol."
CHANGE_COL   = "Change %"
DATE_COL     = "Date"
# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

# remove currency symbols and commas, convert to float
def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip commas and currency symbols, then coerce to float."""
    return (
        series.astype(str)
              .str.replace(r"[,$£€₹]", "", regex=True)
              .str.strip()
              .pipe(pd.to_numeric, errors="coerce")
    )

# remove trailing % sign and convert to float
def _clean_percentage(series: pd.Series) -> pd.Series:
    """Remove trailing '%' and convert to float."""
    return (
        series.astype(str)
              .str.replace("%", "", regex=False)
              .str.strip()
              .pipe(pd.to_numeric, errors="coerce")
    )

# convert K/M/B suffixes to actual numbers (e.g., 1.5M → 1,500,000)
def _parse_volume(series: pd.Series) -> pd.Series:
    """
    Convert Investing.com volume strings to integers.
    Handles suffixes: K (thousands), M (millions), B (billions).
    '-' or blank → NaN.
    """
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

    def _convert(val: str) -> float:
        val = str(val).strip()
        if val in ("-", "", "nan"):
            return float("nan")
        suffix = val[-1].upper()
        if suffix in multipliers:
            return float(val[:-1]) * multipliers[suffix]
        try:
            return float(val.replace(",", ""))
        except ValueError:
            return float("nan")

    return series.map(_convert)

# parse MM/DD/YYYY dates, skips invalid rows
def _safe_parse_date(value: str):
    """Try to parse MM/DD/YYYY; print a warning and return None on failure."""
    try:
        return pd.to_datetime(value, format="%m/%d/%Y")
    except Exception:
        print(f"  [Error] Could not parse date value: {value!r} — skipping row.")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Main Function
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(csv_path: str = CSV_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 · Load raw CSV and print diagnostics
    # ──────────────────────────────────────────────────────────────────────────
    print("Step 1 — Loading Raw Data")
    # Investing.com exports sometimes carry a UTF-8 BOM; encoding_errors='replace'
    # keeps us safe, while skipinitialspace cleans up any leading spaces.
    try:
        raw = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",   # strips BOM if present
            skipinitialspace=True,
        )
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {csv_path}")

    # Strip whitespace from column names (Investing.com)
    raw.columns = raw.columns.str.strip()

    print(f"\n First 5 rows:\n{raw.head()}\n")
    print(f" Last 5 rows:\n{raw.tail()}\n")
    print(" DataFrame info:")
    raw.info()
    print(f"\n Basic statistics (raw, numeric columns only):\n{raw.describe()}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 · Clean the data
    # ──────────────────────────────────────────────────────────────────────────
  
    print("Step 2 — Cleaning Data")

    df = raw.copy()

    # 2-a  Parse Date column (MM/DD/YYYY), row-by-row to catch bad values
    print("\n[2-a] Parsing dates …")
    parsed_dates = df[DATE_COL].map(_safe_parse_date)
    bad_mask = parsed_dates.isna()
    if bad_mask.any():
        print(f"  → Dropping {bad_mask.sum()} row(s) with unparseable dates.")
    df = df[~bad_mask].copy()
    df[DATE_COL] = parsed_dates[~bad_mask]

    # 2-b  Verify expected date range
    print("\n[2-b] Verifying date range …")
    actual_min = df[DATE_COL].min()
    actual_max = df[DATE_COL].max()

    if actual_min > EXPECTED_START:
        warnings.warn(
            f"Data starts at {actual_min.date()} — earlier than expected {EXPECTED_START.date()}."
        )
    if actual_max < EXPECTED_END:
        warnings.warn(
            f"Data ends at {actual_max.date()} — later end expected {EXPECTED_END.date()}."
        )
    print(f"  → Raw date range: {actual_min.date()} → {actual_max.date()}")

    # 2-c  Sort ascending by date
    # sort chronologically for time series analysis
    print("\n[2-c] Sorting by date (ascending) …")
    df.sort_values(DATE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2-d  Report missing values in price columns before conversion
    print("\n[2-d] Missing values in price columns (before conversion):")
    for col in PRICE_COLS:
        if col in df.columns:
            n_miss = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            print(f"  {col}: {n_miss} missing/blank")

    # 2-e  Convert Price, Open, High, Low to float
    print("\n[2-e] Converting price columns to float …")
    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])
        else:
            print(f"  [WARNING] Expected column '{col}' not found — skipping.")

    # 2-f  Handle Vol. column
    # convert volume column (K/M/B suffixes) to numeric
    if VOLUME_COL in df.columns:
        print(f"\n[2-f] Parsing '{VOLUME_COL}' column …")
        df[VOLUME_COL] = _parse_volume(df[VOLUME_COL])
    else:
        print(f"\n[2-f] '{VOLUME_COL}' column not found — skipping.")

    # 2-g  Handle Change % column
    # convert percentage change column to float
    if CHANGE_COL in df.columns:
        print(f"\n[2-g] Parsing '{CHANGE_COL}' column …")
        df[CHANGE_COL] = _clean_percentage(df[CHANGE_COL])
    else:
        print(f"\n[2-g] '{CHANGE_COL}' column not found — skipping.")

    # 2-h  Set Date as index
    # set date as datetime index for time series operations
    print("\n[2-h] Setting Date as index …")
    df.set_index(DATE_COL, inplace=True)

    # 2-i  Feature engineering — Target (next-day close) and Return (%)
    # create target variable (next day price) and daily returns
    print("\n[2-i] Creating feature columns …")
    df["Target"] = df["Price"].shift(-1)   # next trading day's close
    df["Return"] = ((df["Price"] - df["Price"].shift(1)) / df["Price"].shift(1)) * 100

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 · Post-cleaning diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    print("Step 3 — After Cleaning Analysis")

    print(f"\n Date range of cleaned data : {df.index.min().date()} → {df.index.max().date()}")
    print(f" Number of rows             : {len(df)}")

    print(f"\n First 3 rows:\n{df.head(3)}\n")
    print(f" Last 3 rows:\n{df.tail(3)}\n")

    null_counts = df.isna().sum()
    print(" Null counts per column:")
    print(null_counts.to_string())

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 · Saving the data
    # ──────────────────────────────────────────────────────────────────────────
    df.to_csv(output_path)
    print(f"\n Cleaned data saved to: {output_path}")

    return df


# Main Function
if __name__ == "__main__":
    cleaned_df = load_and_clean()
