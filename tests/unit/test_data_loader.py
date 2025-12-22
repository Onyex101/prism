"""
Tests for the DataLoader module.
"""

import pytest
import pandas as pd
from pathlib import Path


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_load_csv_success(self, project_root, tmp_path):
        """Test loading a valid CSV file."""
        from src.data.loader import DataLoader

        # Create test CSV
        test_csv = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df.to_csv(test_csv, index=False)

        # Load
        loader = DataLoader()
        result = loader.load(test_csv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "col1" in result.columns

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        from src.data.loader import DataLoader

        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.csv")

    def test_load_unsupported_format(self, tmp_path):
        """Test loading an unsupported file format."""
        from src.data.loader import DataLoader

        # Create test file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("some content")

        loader = DataLoader()

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(test_file)

    def test_supported_formats(self):
        """Test that supported formats are correctly defined."""
        from src.data.loader import DataLoader

        loader = DataLoader()

        assert ".csv" in loader.SUPPORTED_FORMATS
        assert ".json" in loader.SUPPORTED_FORMATS
        assert ".xlsx" in loader.SUPPORTED_FORMATS
