"""
paths.py
========
Central path configuration for the entire OGDC project.

Import this in every script:
    from paths import PATHS

Then use:
    PATHS["raw"]         → data/raw/
    PATHS["processed"]   → data/processed/
    PATHS["images"]      → outputs/images/
    PATHS["frontend"]    → frontend/public/
    PATHS["scripts"]     → scripts/
"""

import os

# Root of the project — the folder containing this paths.py file
ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    # ── Data ──────────────────────────────────────────────────────────────────
    # Raw:       original downloaded CSV, never modified
    "raw":        os.path.join(ROOT, "data", "raw"),

    # Processed: cleaned CSVs and enriched feature files produced by scripts
    "processed":  os.path.join(ROOT, "data", "processed"),

    # ── Outputs ───────────────────────────────────────────────────────────────
    # Images:   all PNG charts, grouped by script prefix
    "images":     os.path.join(ROOT, "outputs", "images"),

    # ── Frontend ──────────────────────────────────────────────────────────────
    # All PNGs and JSON data files must also land here for the React app
    "frontend":   os.path.join(ROOT, "frontend", "public"),

    # ── Scripts ───────────────────────────────────────────────────────────────
    "scripts":    os.path.join(ROOT, "scripts"),
}


def ensure_dirs():
    """Create all directories if they don't already exist."""
    for key, path in PATHS.items():
        os.makedirs(path, exist_ok=True)


def img(filename):
    """Return full path for an output image file."""
    return os.path.join(PATHS["images"], filename)

def processed(filename):
    """Return full path for a processed data file."""
    return os.path.join(PATHS["processed"], filename)

def raw(filename):
    """Return full path for a raw data file."""
    return os.path.join(PATHS["raw"], filename)

def frontend(filename):
    """Return full path for a frontend public asset."""
    return os.path.join(PATHS["frontend"], filename)