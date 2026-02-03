"""
Loan-Level Feature Engineering Module.

Provides feature transformation and engineering for leveraged loan data.
Designed for use in liquidity prediction models, this module handles:
- Credit rating encoding (ordinal and one-hot)
- Industry sector encoding (one-hot)
- Facility size transformations (log, percentiles)
- Spread normalization (z-scores within rating categories)
- Maturity feature engineering (bins, near-term flags)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class LoanFeatureEngine:
    """
    Feature engineering engine for loan-level data.

    Transforms raw loan data into model-ready features by applying
    standardized encodings, normalizations, and derived calculations.

    Attributes:
        rating_order: Ordered list of credit ratings from best to worst.
        rating_to_ordinal: Mapping of credit ratings to ordinal values.
        sector_categories: List of industry sector categories.

    Example:
        >>> engine = LoanFeatureEngine()
        >>> df_features = engine.transform(df_loans)
    """

    # Credit rating ordering from best (1) to worst (8)
    # Based on typical leveraged loan ratings
    RATING_ORDER: List[str] = [
        'BB+',   # 1 - Best quality for leveraged loans
        'BB',    # 2
        'BB-',   # 3
        'B+',    # 4
        'B',     # 5
        'B-',    # 6
        'CCC+',  # 7
        'CCC',   # 8 - Highest risk
    ]

    # Industry sectors found in leveraged loan market
    SECTOR_CATEGORIES: List[str] = [
        'Consumer',
        'Energy',
        'Financials',
        'Healthcare',
        'Industrials',
        'Technology',
        'Telecom',
        'Utilities',
    ]

    # Maturity bucket definitions (years)
    MATURITY_BINS: List[float] = [0, 2, 4, float('inf')]
    MATURITY_LABELS: List[str] = ['Short', 'Medium', 'Long']

    def __init__(self) -> None:
        """
        Initialize the feature engine.

        Sets up rating-to-ordinal mapping and prepares encoding structures.
        """
        # Create ordinal mapping: BB+ -> 1, BB -> 2, ..., CCC -> 8
        self.rating_to_ordinal: Dict[str, int] = {
            rating: idx + 1 for idx, rating in enumerate(self.RATING_ORDER)
        }

        # Store categories for consistent one-hot encoding
        self.sector_categories = self.SECTOR_CATEGORIES.copy()
        self.rating_categories = self.RATING_ORDER.copy()

        # Track fitted state for percentile calculations
        self._facility_size_values: Optional[np.ndarray] = None

    def encode_credit_rating(self, ratings: pd.Series) -> pd.DataFrame:
        """
        Encode credit ratings using both ordinal and one-hot encoding.

        Ordinal encoding captures the ordered nature of credit risk
        (BB+ < BB < BB- in risk terms). One-hot encoding allows models
        to learn non-linear relationships with specific ratings.

        Args:
            ratings: Series of credit rating strings (e.g., 'BB+', 'B', 'CCC').

        Returns:
            DataFrame with columns:
                - credit_rating_encoded: Ordinal encoding (1-8)
                - rating_BB+, rating_BB, ...: One-hot encoded columns

        Raises:
            ValueError: If unknown rating values are encountered.
        """
        result = pd.DataFrame(index=ratings.index)

        # Ordinal encoding
        ordinal_values = ratings.map(self.rating_to_ordinal)

        # Check for unmapped ratings
        if ordinal_values.isna().any():
            unknown_ratings = ratings[ordinal_values.isna()].unique()
            raise ValueError(f"Unknown credit ratings encountered: {unknown_ratings}")

        result['credit_rating_encoded'] = ordinal_values.astype(int)

        # One-hot encoding with consistent column order
        for rating in self.rating_categories:
            col_name = f'rating_{rating}'
            result[col_name] = (ratings == rating).astype(int)

        return result

    def encode_industry_sector(self, sectors: pd.Series) -> pd.DataFrame:
        """
        One-hot encode industry sectors.

        Creates binary indicator columns for each industry sector,
        enabling models to capture sector-specific effects on liquidity.

        Args:
            sectors: Series of industry sector strings.

        Returns:
            DataFrame with columns sector_Technology, sector_Healthcare, etc.

        Raises:
            ValueError: If unknown sector values are encountered.
        """
        result = pd.DataFrame(index=sectors.index)

        # Check for unknown sectors
        unknown_sectors = set(sectors.unique()) - set(self.sector_categories)
        if unknown_sectors:
            raise ValueError(f"Unknown industry sectors encountered: {unknown_sectors}")

        # One-hot encoding with consistent column order
        for sector in self.sector_categories:
            col_name = f'sector_{sector}'
            result[col_name] = (sectors == sector).astype(int)

        return result

    def calculate_facility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate facility size-related features.

        Transforms raw facility size into normalized features:
        - Log transform to handle right-skewed distribution
        - Percentile rank for relative comparison

        Args:
            df: DataFrame containing 'facility_size' column (in millions).

        Returns:
            DataFrame with columns:
                - facility_size: Original value (millions)
                - facility_size_log: Natural log of facility size
                - facility_size_pctl: Percentile rank (0-100)

        Raises:
            KeyError: If 'facility_size' column is missing.
        """
        if 'facility_size' not in df.columns:
            raise KeyError("DataFrame must contain 'facility_size' column")

        result = pd.DataFrame(index=df.index)

        # Preserve original facility size
        result['facility_size'] = df['facility_size']

        # Log transform (add small constant to handle potential zeros)
        result['facility_size_log'] = np.log(df['facility_size'] + 1)

        # Percentile rank (0-100 scale)
        result['facility_size_pctl'] = df['facility_size'].rank(pct=True) * 100

        # Store values for potential future use
        self._facility_size_values = df['facility_size'].values

        return result

    def calculate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spread-related features.

        Computes z-scores within each rating category to normalize
        spreads relative to peers with similar credit quality.

        Args:
            df: DataFrame containing 'current_spread' and 'credit_rating' columns.

        Returns:
            DataFrame with columns:
                - current_spread: Original spread in basis points
                - spread_z_score: Z-score within rating category

        Raises:
            KeyError: If required columns are missing.
        """
        required_cols = {'current_spread', 'credit_rating'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(f"DataFrame missing required columns: {missing_cols}")

        result = pd.DataFrame(index=df.index)

        # Preserve original spread
        result['current_spread'] = df['current_spread']

        # Calculate z-score within each rating category
        # This normalizes spreads relative to similarly-rated loans
        spread_z_scores = pd.Series(index=df.index, dtype=float)

        for rating in df['credit_rating'].unique():
            mask = df['credit_rating'] == rating
            rating_spreads = df.loc[mask, 'current_spread']

            if len(rating_spreads) > 1:
                mean_spread = rating_spreads.mean()
                std_spread = rating_spreads.std()

                if std_spread > 0:
                    spread_z_scores.loc[mask] = (rating_spreads - mean_spread) / std_spread
                else:
                    # All spreads identical within rating
                    spread_z_scores.loc[mask] = 0.0
            else:
                # Single loan in rating category
                spread_z_scores.loc[mask] = 0.0

        result['spread_z_score'] = spread_z_scores

        return result

    def calculate_maturity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate maturity-related features.

        Creates categorical buckets and near-term maturity flag
        to capture non-linear effects of time to maturity.

        Args:
            df: DataFrame containing 'time_to_maturity' column (in years).

        Returns:
            DataFrame with columns:
                - time_to_maturity: Original value in years
                - maturity_bucket: 'Short' (0-2y), 'Medium' (2-4y), 'Long' (4y+)
                - near_maturity: Boolean flag for loans with < 1 year remaining

        Raises:
            KeyError: If 'time_to_maturity' column is missing.
        """
        if 'time_to_maturity' not in df.columns:
            raise KeyError("DataFrame must contain 'time_to_maturity' column")

        result = pd.DataFrame(index=df.index)

        # Preserve original maturity
        result['time_to_maturity'] = df['time_to_maturity']

        # Categorical bucket: Short/Medium/Long
        result['maturity_bucket'] = pd.cut(
            df['time_to_maturity'],
            bins=self.MATURITY_BINS,
            labels=self.MATURITY_LABELS,
            include_lowest=True
        )

        # Near-maturity flag: loans maturing within 1 year
        result['near_maturity'] = (df['time_to_maturity'] < 1).astype(bool)

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformations to the input DataFrame.

        This is the main entry point for feature engineering. It applies:
        - Credit rating encoding (ordinal + one-hot)
        - Industry sector encoding (one-hot)
        - Facility size features (log, percentile)
        - Spread features (z-score within rating)
        - Maturity features (bucket, near-maturity flag)
        - Covenant lite indicator

        Args:
            df: DataFrame with raw loan data. Expected columns:
                - facility_size: Total facility amount in millions
                - credit_rating: Credit rating string
                - current_spread: Spread over SOFR in basis points
                - time_to_maturity: Years until maturity
                - industry_sector: Industry sector string
                - covenant_lite: Boolean covenant lite indicator

        Returns:
            DataFrame with all engineered features. Original columns
            are preserved alongside new derived features.

        Raises:
            KeyError: If required columns are missing from input.
        """
        # Validate required columns
        required_columns = {
            'facility_size',
            'credit_rating',
            'current_spread',
            'time_to_maturity',
            'industry_sector',
            'covenant_lite',
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Start with empty result DataFrame
        result = pd.DataFrame(index=df.index)

        # Apply facility size features
        facility_features = self.calculate_facility_features(df)
        result = pd.concat([result, facility_features], axis=1)

        # Apply credit rating encoding
        rating_features = self.encode_credit_rating(df['credit_rating'])
        result = pd.concat([result, rating_features], axis=1)

        # Apply spread features (needs credit_rating for z-score calculation)
        spread_features = self.calculate_spread_features(df)
        result = pd.concat([result, spread_features], axis=1)

        # Apply maturity features
        maturity_features = self.calculate_maturity_features(df)
        result = pd.concat([result, maturity_features], axis=1)

        # Apply industry sector encoding
        sector_features = self.encode_industry_sector(df['industry_sector'])
        result = pd.concat([result, sector_features], axis=1)

        # Add covenant lite (ensure boolean type)
        result['covenant_lite'] = df['covenant_lite'].astype(bool)

        # Preserve loan_id if present for tracking
        if 'loan_id' in df.columns:
            result.insert(0, 'loan_id', df['loan_id'])

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names produced by transform().

        Useful for understanding the output schema and for
        feature selection in downstream models.

        Returns:
            List of feature column names.
        """
        features = [
            # Facility features
            'facility_size',
            'facility_size_log',
            'facility_size_pctl',
            # Rating features
            'credit_rating_encoded',
        ]

        # One-hot rating columns
        features.extend([f'rating_{r}' for r in self.rating_categories])

        # Spread features
        features.extend(['current_spread', 'spread_z_score'])

        # Maturity features
        features.extend(['time_to_maturity', 'maturity_bucket', 'near_maturity'])

        # Sector one-hot columns
        features.extend([f'sector_{s}' for s in self.sector_categories])

        # Covenant lite
        features.append('covenant_lite')

        return features

    def get_numeric_feature_names(self) -> List[str]:
        """
        Get list of numeric feature names suitable for ML models.

        Excludes categorical columns (maturity_bucket) that need
        additional encoding for most ML algorithms.

        Returns:
            List of numeric feature column names.
        """
        all_features = self.get_feature_names()

        # Exclude categorical features
        categorical = {'maturity_bucket'}

        return [f for f in all_features if f not in categorical]


def print_feature_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics for engineered features.

    Args:
        df: DataFrame with engineered features.
    """
    print("=" * 70)
    print("LOAN FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(f"\nTotal loans processed: {len(df):,}")
    print(f"Total features: {len(df.columns)}")

    print("\n--- Facility Size Features ---")
    print(f"  facility_size (M):     min={df['facility_size'].min():.0f}, "
          f"max={df['facility_size'].max():.0f}, "
          f"mean={df['facility_size'].mean():.0f}")
    print(f"  facility_size_log:     min={df['facility_size_log'].min():.2f}, "
          f"max={df['facility_size_log'].max():.2f}, "
          f"mean={df['facility_size_log'].mean():.2f}")
    print(f"  facility_size_pctl:    min={df['facility_size_pctl'].min():.1f}, "
          f"max={df['facility_size_pctl'].max():.1f}")

    print("\n--- Credit Rating Features ---")
    rating_cols = [c for c in df.columns if c.startswith('rating_')]
    print(f"  credit_rating_encoded: min={df['credit_rating_encoded'].min()}, "
          f"max={df['credit_rating_encoded'].max()}")
    print(f"  One-hot columns: {len(rating_cols)}")
    for col in rating_cols:
        count = df[col].sum()
        pct = count / len(df) * 100
        print(f"    {col}: {count:,} ({pct:.1f}%)")

    print("\n--- Spread Features ---")
    print(f"  current_spread (bps):  min={df['current_spread'].min():.0f}, "
          f"max={df['current_spread'].max():.0f}, "
          f"mean={df['current_spread'].mean():.0f}")
    print(f"  spread_z_score:        min={df['spread_z_score'].min():.2f}, "
          f"max={df['spread_z_score'].max():.2f}, "
          f"mean={df['spread_z_score'].mean():.2f}")

    print("\n--- Maturity Features ---")
    print(f"  time_to_maturity (y):  min={df['time_to_maturity'].min():.1f}, "
          f"max={df['time_to_maturity'].max():.1f}, "
          f"mean={df['time_to_maturity'].mean():.1f}")
    print(f"  maturity_bucket distribution:")
    bucket_counts = df['maturity_bucket'].value_counts()
    for bucket, count in bucket_counts.items():
        pct = count / len(df) * 100
        print(f"    {bucket}: {count:,} ({pct:.1f}%)")
    near_mat_count = df['near_maturity'].sum()
    print(f"  near_maturity (< 1y):  {near_mat_count:,} ({near_mat_count/len(df)*100:.1f}%)")

    print("\n--- Industry Sector Features ---")
    sector_cols = [c for c in df.columns if c.startswith('sector_')]
    print(f"  One-hot columns: {len(sector_cols)}")
    for col in sorted(sector_cols):
        count = df[col].sum()
        pct = count / len(df) * 100
        print(f"    {col}: {count:,} ({pct:.1f}%)")

    print("\n--- Covenant Lite ---")
    cov_count = df['covenant_lite'].sum()
    print(f"  covenant_lite: {cov_count:,} ({cov_count/len(df)*100:.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    from pathlib import Path

    # Locate data file relative to this script
    script_dir = Path(__file__).parent.parent.parent  # Project root
    data_path = script_dir / "data" / "synthetic_loans.csv"

    print(f"Loading data from: {data_path}")

    # Load synthetic loan data
    df_raw = pd.read_csv(data_path)
    print(f"Loaded {len(df_raw):,} loans")

    # Initialize feature engine
    engine = LoanFeatureEngine()

    # Transform features
    print("\nApplying feature transformations...")
    df_features = engine.transform(df_raw)

    # Print feature summary
    print_feature_summary(df_features)

    # Display sample of transformed data
    print("\n--- Sample Transformed Data (first 5 rows) ---")
    print(df_features.head().to_string())

    # Show feature names
    print("\n--- All Feature Names ---")
    feature_names = engine.get_feature_names()
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")

    print("\n--- Numeric Features (for ML) ---")
    numeric_features = engine.get_numeric_feature_names()
    print(f"  {len(numeric_features)} numeric features available")
