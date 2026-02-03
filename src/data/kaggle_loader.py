"""
Kaggle LendingClub Loan Data Loader.

This module provides functionality to download, load, and preprocess
LendingClub loan data from Kaggle for use in the loan liquidity predictor.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import project utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import get_data_dir, setup_logging

logger = logging.getLogger("loan_liquidity_predictor.kaggle_loader")


class KaggleLoader:
    """
    Loader for LendingClub loan data from Kaggle.

    Handles downloading, loading, preprocessing, and mapping LendingClub data
    to the project's loan schema for use in liquidity prediction models.

    Attributes:
        data_dir: Path to the directory where Kaggle data is stored.
    """

    # LendingClub grade to credit rating mapping
    GRADE_TO_RATING = {
        "A": "BB+",
        "B": "BB",
        "C": "BB-",
        "D": "B+",
        "E": "B",
        "F": "B-",
        "G": "CCC",
    }

    # LendingClub purpose to GICS sector mapping
    PURPOSE_TO_SECTOR = {
        "debt_consolidation": "Financials",
        "credit_card": "Financials",
        "home_improvement": "Consumer Discretionary",
        "major_purchase": "Consumer Discretionary",
        "small_business": "Industrials",
        "car": "Consumer Discretionary",
        "medical": "Health Care",
        "moving": "Real Estate",
        "vacation": "Consumer Discretionary",
        "house": "Real Estate",
        "wedding": "Consumer Discretionary",
        "renewable_energy": "Energy",
        "educational": "Consumer Discretionary",
        "other": "Industrials",
    }

    # Default columns to load (reduces memory usage)
    DEFAULT_COLUMNS = [
        "loan_amnt",
        "int_rate",
        "grade",
        "sub_grade",
        "term",
        "purpose",
        "loan_status",
        "issue_d",
        "emp_length",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "fico_range_low",
        "fico_range_high",
        "open_acc",
        "revol_bal",
        "revol_util",
        "total_acc",
        "home_ownership",
        "verification_status",
    ]

    def __init__(self, data_dir: str = "data/kaggle"):
        """
        Initialize the KaggleLoader.

        Args:
            data_dir: Path to the directory for storing Kaggle data.
                      Can be relative to project root or absolute.
        """
        if Path(data_dir).is_absolute():
            self.data_dir = Path(data_dir)
        else:
            # Relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / data_dir

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"KaggleLoader initialized with data_dir: {self.data_dir}")

    def download_dataset(
        self,
        dataset: str = "wordsforthewise/lending-club",
        force: bool = False
    ) -> bool:
        """
        Download LendingClub dataset from Kaggle.

        Requires Kaggle API credentials to be configured. Falls back to
        using existing local files if download fails.

        Args:
            dataset: Kaggle dataset identifier (owner/dataset-name).
            force: If True, re-download even if files exist locally.

        Returns:
            True if download was successful or files exist locally,
            False otherwise.
        """
        # Check if files already exist
        existing_files = list(self.data_dir.glob("*.csv"))
        if existing_files and not force:
            logger.info(f"Found {len(existing_files)} existing CSV files. "
                       "Use force=True to re-download.")
            return True

        # Check for Kaggle credentials
        kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
        kaggle_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")

        if not kaggle_config.exists() and not kaggle_env:
            logger.warning(
                "Kaggle credentials not found. Please set up Kaggle API:\n"
                "1. Create a Kaggle account at https://www.kaggle.com\n"
                "2. Go to Account Settings -> API -> Create New API Token\n"
                "3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY env vars\n"
                "Falling back to local files."
            )
            if existing_files:
                logger.info(f"Using {len(existing_files)} existing local files.")
                return True
            return False

        try:
            # Import kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                logger.error(
                    "Kaggle package not installed. Install with: pip install kaggle"
                )
                return bool(existing_files)

            # Authenticate and download
            api = KaggleApi()
            api.authenticate()

            logger.info(f"Downloading dataset: {dataset}")
            api.dataset_download_files(
                dataset,
                path=str(self.data_dir),
                unzip=True
            )

            # Verify download
            new_files = list(self.data_dir.glob("*.csv"))
            if new_files:
                logger.info(f"Successfully downloaded {len(new_files)} files.")
                return True
            else:
                logger.warning("Download completed but no CSV files found.")
                return False

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            if existing_files:
                logger.info(f"Falling back to {len(existing_files)} existing local files.")
                return True
            return False

    def load_data(
        self,
        filename: Optional[str] = None,
        columns: Optional[list] = None,
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load LendingClub data from CSV file.

        Args:
            filename: Specific CSV file to load. If None, loads the first
                     CSV file found or the main accepted loans file.
            columns: List of columns to load. If None, loads DEFAULT_COLUMNS.
            nrows: Number of rows to load. If None, loads all rows.

        Returns:
            DataFrame with the loaded loan data.

        Raises:
            FileNotFoundError: If no CSV files are found in data_dir.
        """
        # Find CSV file to load
        if filename:
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
        else:
            # Look for common LendingClub file names
            common_names = [
                "accepted_2007_to_2018Q4.csv",
                "accepted.csv",
                "lending_club_loan_two.csv",
                "loan.csv",
            ]

            filepath = None
            for name in common_names:
                candidate = self.data_dir / name
                if candidate.exists():
                    filepath = candidate
                    break

            # Fall back to first CSV found
            if filepath is None:
                csv_files = list(self.data_dir.glob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError(
                        f"No CSV files found in {self.data_dir}. "
                        "Run download_dataset() first or place files manually."
                    )
                filepath = csv_files[0]

        logger.info(f"Loading data from: {filepath}")

        # Determine columns to load
        use_columns = columns if columns else self.DEFAULT_COLUMNS

        # Load with specified columns if they exist, otherwise load all
        try:
            # First, peek at the file to check available columns
            sample = pd.read_csv(filepath, nrows=5)
            available_columns = set(sample.columns)
            columns_to_load = [c for c in use_columns if c in available_columns]

            if len(columns_to_load) < len(use_columns):
                missing = set(use_columns) - set(columns_to_load)
                logger.warning(f"Columns not found in file: {missing}")

            if not columns_to_load:
                logger.warning("None of the requested columns found. Loading all columns.")
                columns_to_load = None

            df = pd.read_csv(
                filepath,
                usecols=columns_to_load,
                nrows=nrows,
                low_memory=False
            )

        except Exception as e:
            logger.warning(f"Error loading with column filter: {e}. Loading all columns.")
            df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        drop_threshold: float = 0.5,
        filter_non_default: bool = True
    ) -> pd.DataFrame:
        """
        Clean and prepare LendingClub data for modeling.

        Preprocessing steps:
        1. Remove duplicates
        2. Drop rows with excessive missing values
        3. Impute remaining missing values
        4. Filter to relevant loan statuses
        5. Convert data types
        6. Encode categorical variables

        Args:
            df: Raw DataFrame from load_data().
            drop_threshold: Fraction of missing values above which to drop rows.
            filter_non_default: If True, filter to non-defaulted loans for training.

        Returns:
            Preprocessed DataFrame ready for feature engineering.
        """
        logger.info(f"Preprocessing {len(df)} rows...")
        df = df.copy()

        # Step 1: Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Step 2: Drop rows with excessive missing values
        row_missing_frac = df.isna().sum(axis=1) / len(df.columns)
        df = df[row_missing_frac <= drop_threshold]
        logger.info(f"After dropping high-missing rows: {len(df)} rows remain")

        # Step 3: Filter loan status if requested
        if filter_non_default and "loan_status" in df.columns:
            # Keep only current and fully paid loans for training
            valid_statuses = [
                "Fully Paid",
                "Current",
                "In Grace Period",
            ]
            mask = df["loan_status"].isin(valid_statuses)
            df = df[mask]
            logger.info(f"After filtering to non-default loans: {len(df)} rows")

        # Step 4: Convert data types and clean columns

        # Convert interest rate from string to float (remove '%')
        if "int_rate" in df.columns:
            if df["int_rate"].dtype == "object":
                df["int_rate"] = (
                    df["int_rate"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .astype(float)
                )

        # Convert term from string to integer (extract months)
        if "term" in df.columns:
            if df["term"].dtype == "object":
                df["term_months"] = (
                    df["term"]
                    .astype(str)
                    .str.extract(r"(\d+)")
                    .astype(float)
                )

        # Convert employment length to numeric
        if "emp_length" in df.columns:
            emp_map = {
                "< 1 year": 0.5,
                "1 year": 1,
                "2 years": 2,
                "3 years": 3,
                "4 years": 4,
                "5 years": 5,
                "6 years": 6,
                "7 years": 7,
                "8 years": 8,
                "9 years": 9,
                "10+ years": 10,
            }
            df["emp_length_years"] = df["emp_length"].map(emp_map)

        # Convert revol_util from string to float
        if "revol_util" in df.columns:
            if df["revol_util"].dtype == "object":
                df["revol_util"] = (
                    df["revol_util"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .replace("nan", np.nan)
                    .astype(float)
                )

        # Step 5: Impute missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Numeric: impute with median
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # Categorical: impute with mode
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])

        # Step 6: Encode categorical variables
        # Create encoded versions while keeping originals for mapping
        if "grade" in df.columns:
            df["grade_encoded"] = df["grade"].astype("category").cat.codes

        if "sub_grade" in df.columns:
            df["sub_grade_encoded"] = df["sub_grade"].astype("category").cat.codes

        if "home_ownership" in df.columns:
            df["home_ownership_encoded"] = pd.get_dummies(
                df["home_ownership"], prefix="home", drop_first=True
            ).values.argmax(axis=1) if len(df) > 0 else 0

        if "verification_status" in df.columns:
            df["verification_encoded"] = (
                df["verification_status"]
                .map({
                    "Not Verified": 0,
                    "Source Verified": 1,
                    "Verified": 2
                })
                .fillna(0)
            )

        logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def map_to_loan_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map LendingClub columns to the project's loan schema.

        Mapping:
        - loan_amnt -> facility_size (converted to millions)
        - int_rate -> current_spread (converted to basis points)
        - grade/sub_grade -> credit_rating (letter grade to rating)
        - term -> time_to_maturity (months to years)
        - purpose -> industry_sector (mapped to GICS sectors)

        Args:
            df: Preprocessed DataFrame from preprocess().

        Returns:
            DataFrame with columns mapped to project schema.
        """
        logger.info("Mapping columns to project loan schema...")
        result = pd.DataFrame()

        # loan_amnt -> facility_size (convert to millions)
        if "loan_amnt" in df.columns:
            result["facility_size"] = df["loan_amnt"] / 1_000_000
            logger.debug("Mapped loan_amnt -> facility_size (in millions)")

        # int_rate -> current_spread (convert percentage to basis points)
        # Note: int_rate is already a percentage (e.g., 10.5 means 10.5%)
        # Convert to basis points: multiply by 100
        if "int_rate" in df.columns:
            result["current_spread"] = df["int_rate"] * 100
            logger.debug("Mapped int_rate -> current_spread (in bps)")

        # grade/sub_grade -> credit_rating
        if "grade" in df.columns:
            result["credit_rating"] = df["grade"].map(self.GRADE_TO_RATING)
            # Fill any unmapped grades with 'B'
            result["credit_rating"] = result["credit_rating"].fillna("B")
            logger.debug("Mapped grade -> credit_rating")

        # term -> time_to_maturity (convert months to years)
        if "term_months" in df.columns:
            result["time_to_maturity"] = df["term_months"] / 12
        elif "term" in df.columns:
            # Extract months from string if not already processed
            months = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
            result["time_to_maturity"] = months / 12
        logger.debug("Mapped term -> time_to_maturity (in years)")

        # purpose -> industry_sector
        if "purpose" in df.columns:
            result["industry_sector"] = df["purpose"].map(self.PURPOSE_TO_SECTOR)
            # Fill any unmapped purposes with 'Industrials'
            result["industry_sector"] = result["industry_sector"].fillna("Industrials")
            logger.debug("Mapped purpose -> industry_sector")

        # Preserve useful features for modeling
        preserve_columns = [
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
            "open_acc",
            "revol_bal",
            "revol_util",
            "total_acc",
            "emp_length_years",
            "verification_encoded",
            "home_ownership_encoded",
            "loan_status",
            "issue_d",
        ]

        for col in preserve_columns:
            if col in df.columns:
                result[col] = df[col]

        # Create derived features
        if "fico_range_low" in result.columns and "fico_range_high" in result.columns:
            result["fico_score"] = (
                result["fico_range_low"] + result["fico_range_high"]
            ) / 2
            result = result.drop(columns=["fico_range_low", "fico_range_high"])

        logger.info(f"Schema mapping complete: {len(result.columns)} columns")
        logger.info(f"Output columns: {list(result.columns)}")
        return result

    def load_and_process(
        self,
        filename: Optional[str] = None,
        nrows: Optional[int] = None,
        download: bool = True
    ) -> pd.DataFrame:
        """
        Convenience method to run the full pipeline.

        Args:
            filename: Specific CSV file to load.
            nrows: Number of rows to load.
            download: Whether to attempt download if files don't exist.

        Returns:
            Fully processed DataFrame ready for modeling.
        """
        if download:
            self.download_dataset()

        df = self.load_data(filename=filename, nrows=nrows)
        df = self.preprocess(df)
        df = self.map_to_loan_schema(df)

        return df


if __name__ == "__main__":
    """
    Demonstration of the full Kaggle LendingClub data loading pipeline.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("LendingClub Kaggle Data Loader - Demo Pipeline")
    print("=" * 60)

    # Initialize loader
    loader = KaggleLoader()
    print(f"\nData directory: {loader.data_dir}")

    # Step 1: Download (will use existing files or warn about credentials)
    print("\n--- Step 1: Download Dataset ---")
    download_success = loader.download_dataset()
    print(f"Download status: {'Success' if download_success else 'Failed/Skipped'}")

    # Step 2: Load data (limit rows for demo)
    print("\n--- Step 2: Load Data ---")
    try:
        df_raw = loader.load_data(nrows=10000)
        print(f"Loaded shape: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")
        print(f"\nSample data:")
        print(df_raw.head(3).to_string())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo use this loader, either:")
        print("1. Set up Kaggle API credentials and run download_dataset()")
        print("2. Manually download LendingClub data and place CSV in data/kaggle/")
        sys.exit(1)

    # Step 3: Preprocess
    print("\n--- Step 3: Preprocess ---")
    df_processed = loader.preprocess(df_raw)
    print(f"Processed shape: {df_processed.shape}")
    print(f"Missing values per column:")
    missing = df_processed.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    # Step 4: Map to schema
    print("\n--- Step 4: Map to Loan Schema ---")
    df_final = loader.map_to_loan_schema(df_processed)
    print(f"Final shape: {df_final.shape}")
    print(f"Schema columns: {list(df_final.columns)}")
    print(f"\nSample mapped data:")
    print(df_final.head(5).to_string())

    # Summary statistics
    print("\n--- Summary Statistics ---")
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    print(df_final[numeric_cols].describe().to_string())

    print("\n--- Credit Rating Distribution ---")
    if "credit_rating" in df_final.columns:
        print(df_final["credit_rating"].value_counts())

    print("\n--- Industry Sector Distribution ---")
    if "industry_sector" in df_final.columns:
        print(df_final["industry_sector"].value_counts())

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
