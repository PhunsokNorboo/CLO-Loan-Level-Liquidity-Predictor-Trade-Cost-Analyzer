"""
Market-level feature engineering module.

This module provides the MarketFeatureEngine class for calculating
market-level features used in loan liquidity prediction, including:
- VIX volatility features and regime classification
- Credit spread features (HY and IG)
- Interest rate features (Fed funds and yield curve)
- Composite market stress indicator

The features capture macroeconomic conditions that influence
loan trading activity and liquidity.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketFeatureEngine:
    """
    A class for calculating market-level features for loan liquidity prediction.

    This engine transforms raw market data (VIX, credit spreads, interest rates)
    into features that capture market conditions affecting loan liquidity.
    Features include levels, changes, percentiles, regimes, and a composite
    stress indicator.

    Attributes:
        vix_percentile_window (int): Rolling window for VIX percentile calculation.
        fed_funds_change_window (int): Window for Fed funds rate change calculation.
        stress_weights (dict): Weights for stress indicator components.

    Example:
        >>> engine = MarketFeatureEngine()
        >>> market_df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=100),
        ...     'vix': np.random.uniform(10, 40, 100),
        ...     'hy_spread': np.random.uniform(3, 8, 100),
        ...     'ig_spread': np.random.uniform(1, 3, 100),
        ...     'fed_funds_rate': np.random.uniform(4, 5.5, 100),
        ...     'yield_curve_slope': np.random.uniform(-1, 2, 100)
        ... })
        >>> features_df = engine.transform(market_df)
    """

    # VIX regime thresholds (based on historical analysis)
    VIX_LOW_THRESHOLD = 15.0
    VIX_NORMAL_THRESHOLD = 20.0
    VIX_HIGH_THRESHOLD = 30.0

    # Default stress indicator weights
    DEFAULT_STRESS_WEIGHTS = {
        'vix': 0.4,
        'hy_spread': 0.4,
        'curve_inversion': 0.2
    }

    def __init__(
        self,
        vix_percentile_window: int = 252,
        fed_funds_change_window: int = 30,
        stress_weights: Optional[dict] = None
    ):
        """
        Initialize the MarketFeatureEngine.

        Args:
            vix_percentile_window: Rolling window (in days) for VIX percentile
                calculation. Default is 252 (approximately one trading year).
            fed_funds_change_window: Window (in days) for calculating Fed funds
                rate changes. Default is 30 days.
            stress_weights: Dictionary with weights for stress indicator components.
                Keys: 'vix', 'hy_spread', 'curve_inversion'. Values should sum to 1.
                If None, uses default weights (0.4, 0.4, 0.2).
        """
        self.vix_percentile_window = vix_percentile_window
        self.fed_funds_change_window = fed_funds_change_window

        # Set stress weights (validate they sum to 1)
        if stress_weights is None:
            self.stress_weights = self.DEFAULT_STRESS_WEIGHTS.copy()
        else:
            weight_sum = sum(stress_weights.values())
            if not np.isclose(weight_sum, 1.0):
                logger.warning(
                    f"Stress weights sum to {weight_sum}, normalizing to 1.0"
                )
                self.stress_weights = {
                    k: v / weight_sum for k, v in stress_weights.items()
                }
            else:
                self.stress_weights = stress_weights.copy()

        logger.info(
            f"MarketFeatureEngine initialized with "
            f"vix_percentile_window={vix_percentile_window}, "
            f"fed_funds_change_window={fed_funds_change_window}"
        )

    def calculate_vix_features(self, vix_data: pd.Series) -> pd.DataFrame:
        """
        Calculate VIX-based volatility features.

        Computes VIX level, rolling percentile rank, and volatility regime
        classification based on historical thresholds.

        Args:
            vix_data: Pandas Series containing VIX index values.

        Returns:
            DataFrame with columns:
                - vix_level: Raw VIX index value
                - vix_percentile: Rolling percentile rank (0-100)
                - vix_regime: Categorical regime ('Low', 'Normal', 'High', 'Extreme')

        Example:
            >>> engine = MarketFeatureEngine()
            >>> vix = pd.Series([15.2, 18.5, 22.1, 35.6, 28.3])
            >>> features = engine.calculate_vix_features(vix)
            >>> print(features['vix_regime'].value_counts())
        """
        result = pd.DataFrame()

        # VIX level (raw value)
        result['vix_level'] = vix_data.values

        # Rolling percentile rank
        # For each observation, calculate what percentile it falls in
        # within the rolling window
        result['vix_percentile'] = vix_data.rolling(
            window=self.vix_percentile_window,
            min_periods=1
        ).apply(
            lambda x: 100 * (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1),
            raw=False
        ).values

        # VIX regime classification
        def classify_vix_regime(vix_value: float) -> str:
            """Classify VIX into volatility regime."""
            if pd.isna(vix_value):
                return np.nan
            elif vix_value < self.VIX_LOW_THRESHOLD:
                return 'Low'
            elif vix_value < self.VIX_NORMAL_THRESHOLD:
                return 'Normal'
            elif vix_value < self.VIX_HIGH_THRESHOLD:
                return 'High'
            else:
                return 'Extreme'

        result['vix_regime'] = vix_data.apply(classify_vix_regime)

        logger.debug(f"Calculated VIX features for {len(result)} observations")
        return result

    def calculate_spread_features(
        self,
        hy_spread: pd.Series,
        ig_spread: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate credit spread features.

        Computes high yield and investment grade spread levels,
        plus the HY-IG spread gap which indicates credit risk differentiation.

        Args:
            hy_spread: Pandas Series with high yield corporate bond spreads.
            ig_spread: Pandas Series with investment grade corporate bond spreads.

        Returns:
            DataFrame with columns:
                - hy_spread: High yield credit spread level
                - ig_spread: Investment grade credit spread level
                - hy_ig_gap: Difference between HY and IG spreads

        Example:
            >>> engine = MarketFeatureEngine()
            >>> hy = pd.Series([4.5, 5.2, 6.1, 7.8, 5.5])
            >>> ig = pd.Series([1.2, 1.4, 1.8, 2.5, 1.6])
            >>> features = engine.calculate_spread_features(hy, ig)
            >>> print(features['hy_ig_gap'].mean())
        """
        result = pd.DataFrame()

        # Spread levels
        result['hy_spread'] = hy_spread.values
        result['ig_spread'] = ig_spread.values

        # HY-IG gap (spread differentiation)
        # Higher gap indicates greater credit risk differentiation
        result['hy_ig_gap'] = hy_spread.values - ig_spread.values

        logger.debug(f"Calculated spread features for {len(result)} observations")
        return result

    def calculate_rate_features(
        self,
        fed_funds: pd.Series,
        yield_curve: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate interest rate features.

        Computes Fed funds rate level and changes, yield curve slope,
        and curve inversion indicator.

        Args:
            fed_funds: Pandas Series with Federal funds effective rate.
            yield_curve: Pandas Series with yield curve slope (10Y-2Y spread).

        Returns:
            DataFrame with columns:
                - fed_funds_rate: Current Fed funds rate level
                - fed_funds_change_30d: 30-day change in Fed funds rate
                - yield_curve_slope: 10Y-2Y Treasury spread
                - curve_inverted: Boolean flag (True if slope < 0)

        Example:
            >>> engine = MarketFeatureEngine()
            >>> fed = pd.Series([5.25, 5.25, 5.50, 5.50, 5.50])
            >>> yc = pd.Series([0.5, 0.3, 0.1, -0.2, -0.5])
            >>> features = engine.calculate_rate_features(fed, yc)
            >>> print(features['curve_inverted'].sum())
        """
        result = pd.DataFrame()

        # Fed funds level
        result['fed_funds_rate'] = fed_funds.values

        # Fed funds rate change over window
        fed_funds_series = pd.Series(fed_funds.values)
        result['fed_funds_change_30d'] = fed_funds_series.diff(
            periods=self.fed_funds_change_window
        ).values

        # Yield curve slope
        result['yield_curve_slope'] = yield_curve.values

        # Curve inversion indicator
        result['curve_inverted'] = (yield_curve.values < 0).astype(bool)

        logger.debug(f"Calculated rate features for {len(result)} observations")
        return result

    def calculate_stress_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite market stress indicator.

        Combines normalized VIX, HY spread, and yield curve inversion
        into a single stress score ranging from 0 (low stress) to 1 (high stress).

        The normalization uses min-max scaling based on typical ranges:
        - VIX: 10-80 range
        - HY spread: 2-20 range
        - Curve inversion: binary (0 or 1)

        Args:
            df: DataFrame containing at minimum:
                - 'vix_level' or 'vix': VIX index value
                - 'hy_spread': High yield spread
                - 'yield_curve_slope': Yield curve slope OR
                - 'curve_inverted': Boolean inversion indicator

        Returns:
            Series with composite stress indicator values (0-1 scale).

        Example:
            >>> engine = MarketFeatureEngine()
            >>> df = pd.DataFrame({
            ...     'vix_level': [15, 25, 40],
            ...     'hy_spread': [3.5, 5.5, 10.0],
            ...     'yield_curve_slope': [1.0, -0.5, -1.5]
            ... })
            >>> stress = engine.calculate_stress_indicator(df)
            >>> print(stress)
        """
        # Define normalization ranges (based on historical analysis)
        VIX_MIN, VIX_MAX = 10.0, 80.0
        HY_SPREAD_MIN, HY_SPREAD_MAX = 2.0, 20.0

        # Get VIX values (handle both column name variants)
        vix_col = 'vix_level' if 'vix_level' in df.columns else 'vix'
        if vix_col not in df.columns:
            raise ValueError("DataFrame must contain 'vix_level' or 'vix' column")

        vix_values = df[vix_col].values

        # Normalize VIX to 0-1
        vix_normalized = np.clip(
            (vix_values - VIX_MIN) / (VIX_MAX - VIX_MIN),
            0.0, 1.0
        )

        # Get HY spread
        if 'hy_spread' not in df.columns:
            raise ValueError("DataFrame must contain 'hy_spread' column")

        hy_values = df['hy_spread'].values

        # Normalize HY spread to 0-1
        hy_normalized = np.clip(
            (hy_values - HY_SPREAD_MIN) / (HY_SPREAD_MAX - HY_SPREAD_MIN),
            0.0, 1.0
        )

        # Get curve inversion indicator
        if 'curve_inverted' in df.columns:
            curve_inverted = df['curve_inverted'].astype(float).values
        elif 'yield_curve_slope' in df.columns:
            curve_inverted = (df['yield_curve_slope'].values < 0).astype(float)
        else:
            raise ValueError(
                "DataFrame must contain 'curve_inverted' or 'yield_curve_slope' column"
            )

        # Calculate weighted stress indicator
        stress = (
            self.stress_weights['vix'] * vix_normalized +
            self.stress_weights['hy_spread'] * hy_normalized +
            self.stress_weights['curve_inversion'] * curve_inverted
        )

        # Ensure result is in 0-1 range
        stress = np.clip(stress, 0.0, 1.0)

        logger.debug(
            f"Calculated stress indicator: mean={np.nanmean(stress):.3f}, "
            f"max={np.nanmax(stress):.3f}"
        )

        return pd.Series(stress, name='market_stress')

    def transform(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all market feature transformations.

        This is the main entry point for feature engineering. It applies
        all individual feature calculations and returns a consolidated
        DataFrame with all market features.

        Args:
            market_df: DataFrame with market data containing columns:
                - 'date': Date column (optional, will be preserved if present)
                - 'vix': VIX index values
                - 'hy_spread': High yield spread
                - 'ig_spread': Investment grade spread
                - 'fed_funds_rate': Federal funds rate
                - 'yield_curve_slope': 10Y-2Y Treasury spread

        Returns:
            DataFrame with all engineered features:
                - date (if present in input)
                - vix_level
                - vix_percentile
                - vix_regime
                - hy_spread
                - ig_spread
                - hy_ig_gap
                - fed_funds_rate
                - fed_funds_change_30d
                - yield_curve_slope
                - curve_inverted
                - market_stress

        Raises:
            ValueError: If required columns are missing from input DataFrame.

        Example:
            >>> engine = MarketFeatureEngine()
            >>> raw_data = fetcher.fetch_all_indicators('2020-01-01', '2023-12-31')
            >>> features = engine.transform(raw_data)
            >>> print(features.columns.tolist())
        """
        logger.info(f"Transforming market data with {len(market_df)} observations")

        # Validate required columns
        required_cols = ['vix', 'hy_spread', 'ig_spread', 'fed_funds_rate', 'yield_curve_slope']
        missing_cols = [col for col in required_cols if col not in market_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Initialize result DataFrame
        result = pd.DataFrame()

        # Preserve date column if present
        if 'date' in market_df.columns:
            result['date'] = market_df['date'].values

        # Calculate VIX features
        vix_features = self.calculate_vix_features(market_df['vix'])
        for col in vix_features.columns:
            result[col] = vix_features[col].values

        # Calculate spread features
        spread_features = self.calculate_spread_features(
            market_df['hy_spread'],
            market_df['ig_spread']
        )
        for col in spread_features.columns:
            result[col] = spread_features[col].values

        # Calculate rate features
        rate_features = self.calculate_rate_features(
            market_df['fed_funds_rate'],
            market_df['yield_curve_slope']
        )
        for col in rate_features.columns:
            result[col] = rate_features[col].values

        # Calculate composite stress indicator
        result['market_stress'] = self.calculate_stress_indicator(result).values

        logger.info(
            f"Transformation complete: {len(result)} rows, "
            f"{len(result.columns)} features"
        )

        return result


if __name__ == "__main__":
    """
    Demonstration of MarketFeatureEngine usage.

    This example shows how to:
    1. Initialize the feature engine
    2. Create sample market data
    3. Apply transformations
    4. Examine the resulting features
    """
    print("=" * 70)
    print("Market Feature Engineering - Demo")
    print("=" * 70)

    # Create sample market data
    print("\n1. Creating sample market data...")
    np.random.seed(42)
    n_days = 365

    # Generate realistic sample data
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # VIX: typically 10-30, with occasional spikes
    vix_base = 18 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_days))
    vix_noise = np.random.normal(0, 3, n_days)
    vix = np.clip(vix_base + vix_noise, 10, 60)

    # HY spread: typically 3-6%, inversely correlated with market conditions
    hy_spread_base = 4.5 + 0.05 * (vix - 20)
    hy_spread = np.clip(hy_spread_base + np.random.normal(0, 0.3, n_days), 2.5, 12)

    # IG spread: lower and more stable than HY
    ig_spread = np.clip(hy_spread * 0.3 + np.random.normal(0, 0.1, n_days), 0.8, 4)

    # Fed funds rate: relatively stable with gradual changes
    fed_funds = np.clip(
        5.25 + 0.25 * np.cumsum(np.random.choice([-0.01, 0, 0.01], n_days)),
        3.0, 6.0
    )

    # Yield curve slope: can be positive or negative
    yield_curve = np.clip(
        0.5 - 0.02 * (vix - 20) + np.random.normal(0, 0.3, n_days),
        -2.0, 3.0
    )

    market_df = pd.DataFrame({
        'date': dates,
        'vix': vix,
        'hy_spread': hy_spread,
        'ig_spread': ig_spread,
        'fed_funds_rate': fed_funds,
        'yield_curve_slope': yield_curve
    })

    print(f"   Created {len(market_df)} days of market data")
    print("\n   Sample raw data (first 5 rows):")
    print(market_df.head().to_string(index=False))

    # Initialize the feature engine
    print("\n2. Initializing MarketFeatureEngine...")
    engine = MarketFeatureEngine(
        vix_percentile_window=252,
        fed_funds_change_window=30
    )
    print("   Engine initialized successfully!")

    # Transform the data
    print("\n3. Applying transformations...")
    features_df = engine.transform(market_df)

    print(f"\n   Feature DataFrame shape: {features_df.shape}")
    print("\n   Engineered features:")
    for col in features_df.columns:
        print(f"      - {col}")

    # Display sample output
    print("\n   Sample transformed data (first 5 rows):")
    display_cols = ['date', 'vix_level', 'vix_regime', 'hy_ig_gap',
                    'curve_inverted', 'market_stress']
    print(features_df[display_cols].head().to_string(index=False))

    # Show VIX regime distribution
    print("\n4. Feature Analysis:")
    print("-" * 70)

    print("\n   VIX Regime Distribution:")
    regime_counts = features_df['vix_regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(features_df)
        print(f"      {regime}: {count} days ({pct:.1f}%)")

    # Show yield curve inversion stats
    inverted_days = features_df['curve_inverted'].sum()
    print(f"\n   Yield Curve Inversion:")
    print(f"      Inverted days: {inverted_days} ({100*inverted_days/len(features_df):.1f}%)")

    # Show stress indicator statistics
    print("\n   Market Stress Indicator:")
    stress = features_df['market_stress']
    print(f"      Mean:   {stress.mean():.3f}")
    print(f"      Std:    {stress.std():.3f}")
    print(f"      Min:    {stress.min():.3f}")
    print(f"      Max:    {stress.max():.3f}")
    print(f"      Median: {stress.median():.3f}")

    # Show correlation between stress and VIX
    print("\n   Stress Indicator Correlations:")
    print(f"      Stress vs VIX:       {stress.corr(features_df['vix_level']):.3f}")
    print(f"      Stress vs HY Spread: {stress.corr(features_df['hy_spread']):.3f}")
    print(f"      Stress vs HY-IG Gap: {stress.corr(features_df['hy_ig_gap']):.3f}")

    # Identify high stress periods
    high_stress_threshold = 0.6
    high_stress_days = features_df[features_df['market_stress'] > high_stress_threshold]
    print(f"\n   High Stress Periods (stress > {high_stress_threshold}):")
    print(f"      {len(high_stress_days)} days ({100*len(high_stress_days)/len(features_df):.1f}%)")

    if len(high_stress_days) > 0:
        print("\n   Sample high stress days:")
        sample_cols = ['date', 'vix_level', 'vix_regime', 'hy_spread', 'market_stress']
        print(high_stress_days[sample_cols].head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
