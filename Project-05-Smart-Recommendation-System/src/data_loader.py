"""Utilities for loading raw recommendation data."""

from pathlib import Path


def load_csv(path: str | Path):
    """Load a CSV file using pandas."""
    import pandas as pd

    return pd.read_csv(path)
