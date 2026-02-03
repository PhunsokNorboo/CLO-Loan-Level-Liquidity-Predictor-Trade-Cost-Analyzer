"""
Liquidity Feature Engineering Module.

Provides feature engineering for liquidity indicators in leveraged loan markets.
Calculates volume metrics, bid-ask spreads, trading recency, dealer activity,
and ownership concentration features.
"""

import numpy as np
import pandas as pd
from typing import Optional


class LiquidityFeatureEngine:
    """
    Feature engineering engine for liquidity indicators.

    Calculates a comprehensive set of liquidity-related features that capture:
    - Trading volume and turnover
    - Bid-ask spread dynamics
    - Trading recency and frequency
    - Dealer participation and coverage
    - CLO ownership and concentration

    These features help predict liquidity tiers and trading conditions
    for leveraged loans.
    """

    # Feature metadata for documentation
    FEATURE_SPECS = {
        'trading_volume_30d': 'Rolling 30-day trading volume',
        'volume_percentile': 'Percentile rank of volume',
        'volume_to_size_ratio': 'Turnover ratio',
        'price_volatility_30d': 'Rolling 30-day price std dev',
        'bid_ask_spread': 'Average bid-ask spread',
        'bid_ask_percentile': 'Percentile of spread',
        'spread_volatility': 'Spread standard deviation',
        'dealer_quote_count': 'Number of dealers quoting',
        'dealer_coverage': 'Quote count / max dealers',
        'days_since_last_trade': 'Recency of trading activity',
        'trade_frequency': 'Trades per week',
        'clo_ownership_pct': 'Percentage held by CLOs',
        'ownership_concentration': 'HHI of top holders',
    }

    def __init__(self, max_dealers: int = 50):
        """
        Initialize the LiquidityFeatureEngine.

        Args:
            max_dealers: Maximum number of dealers in the market,
                         used for dealer coverage calculation.
        """
        self.max_dealers = max_dealers

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-related liquidity features.

        Features calculated:
        - trading_volume_30d: 30-day rolling trading volume
        - volume_percentile: Percentile rank of volume across loans
        - volume_to_size_ratio: Turnover ratio (volume / facility size)
        - price_volatility_30d: Rolling price volatility

        Args:
            df: DataFrame containing loan data with columns:
                - trading_volume_30d (or raw trade data)
                - facility_size
                - price or current_price (optional, for volatility)

        Returns:
            DataFrame with volume features added.
        """
        result = df.copy()

        # Ensure trading_volume_30d exists
        if 'trading_volume_30d' not in result.columns:
            result['trading_volume_30d'] = np.nan

        # Calculate volume percentile (rank-based)
        result['volume_percentile'] = result['trading_volume_30d'].rank(
            pct=True, method='average'
        ) * 100

        # Calculate volume-to-size ratio (turnover)
        if 'facility_size' in result.columns:
            # Avoid division by zero
            result['volume_to_size_ratio'] = np.where(
                result['facility_size'] > 0,
                result['trading_volume_30d'] / result['facility_size'],
                0.0
            )
        else:
            result['volume_to_size_ratio'] = np.nan

        # Calculate price volatility if price history available
        # For single-point data, use proxy from bid-ask spread
        if 'price_volatility_30d' not in result.columns:
            if 'bid_ask_spread' in result.columns:
                # Proxy: higher bid-ask spreads correlate with price volatility
                # Scale factor calibrated to typical leveraged loan volatility
                result['price_volatility_30d'] = result['bid_ask_spread'] * 0.02
            else:
                result['price_volatility_30d'] = np.nan

        return result

    def calculate_bid_ask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bid-ask spread related features.

        Features calculated:
        - bid_ask_spread: Average bid-ask spread in basis points
        - bid_ask_percentile: Percentile rank of spread
        - spread_volatility: Standard deviation of spread

        Args:
            df: DataFrame containing:
                - bid_ask_spread (or bid and ask columns)

        Returns:
            DataFrame with bid-ask features added.
        """
        result = df.copy()

        # Calculate bid-ask spread if not present but bid/ask available
        if 'bid_ask_spread' not in result.columns:
            if 'bid' in result.columns and 'ask' in result.columns:
                result['bid_ask_spread'] = result['ask'] - result['bid']
            else:
                result['bid_ask_spread'] = np.nan

        # Calculate spread percentile (higher percentile = wider spread = less liquid)
        result['bid_ask_percentile'] = result['bid_ask_spread'].rank(
            pct=True, method='average'
        ) * 100

        # Calculate spread volatility
        # For cross-sectional data, use sector-based proxy
        if 'spread_volatility' not in result.columns:
            if 'industry_sector' in result.columns:
                # Calculate within-sector spread variation as proxy
                sector_std = result.groupby('industry_sector')['bid_ask_spread'].transform('std')
                result['spread_volatility'] = sector_std.fillna(
                    result['bid_ask_spread'].std()
                )
            else:
                # Use cross-sectional standard deviation as baseline
                result['spread_volatility'] = result['bid_ask_spread'] * 0.15

        return result

    def calculate_trading_recency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading recency and frequency features.

        Features calculated:
        - days_since_last_trade: Days since most recent trade
        - trade_frequency: Average trades per week

        Args:
            df: DataFrame containing:
                - last_trade_date (optional)
                - trade_count_30d or trading_volume_30d

        Returns:
            DataFrame with trading recency features added.
        """
        result = df.copy()

        # Calculate days since last trade
        if 'days_since_last_trade' not in result.columns:
            if 'last_trade_date' in result.columns:
                reference_date = pd.Timestamp.now()
                result['days_since_last_trade'] = (
                    reference_date - pd.to_datetime(result['last_trade_date'])
                ).dt.days
            else:
                # Estimate from volume: lower volume -> less frequent trading
                # Assume high volume loans trade daily, low volume monthly
                if 'trading_volume_30d' in result.columns:
                    volume_normalized = result['trading_volume_30d'].rank(pct=True)
                    # High volume (percentile 1.0) -> 1 day, low volume (0.0) -> 30 days
                    result['days_since_last_trade'] = np.round(
                        1 + (1 - volume_normalized) * 29
                    )
                else:
                    result['days_since_last_trade'] = np.nan

        # Calculate trade frequency (trades per week)
        if 'trade_frequency' not in result.columns:
            if 'trade_count_30d' in result.columns:
                # 30 days ~ 4.3 weeks
                result['trade_frequency'] = result['trade_count_30d'] / 4.3
            elif 'trading_volume_30d' in result.columns and 'facility_size' in result.columns:
                # Estimate: assume average trade size is 1% of facility
                avg_trade_size = result['facility_size'] * 0.01
                estimated_trades = result['trading_volume_30d'] / avg_trade_size.clip(lower=0.1)
                result['trade_frequency'] = estimated_trades / 4.3
            else:
                result['trade_frequency'] = np.nan

        return result

    def calculate_dealer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dealer activity metrics.

        Features calculated:
        - dealer_quote_count: Number of dealers actively quoting
        - dealer_coverage: Quote count as percentage of max dealers

        Args:
            df: DataFrame containing:
                - dealer_quote_count (optional)
                - dealer_ids or similar (optional)

        Returns:
            DataFrame with dealer metrics added.
        """
        result = df.copy()

        # Calculate dealer quote count if not present
        if 'dealer_quote_count' not in result.columns:
            if 'dealer_ids' in result.columns:
                # Count unique dealers
                result['dealer_quote_count'] = result['dealer_ids'].apply(
                    lambda x: len(x) if isinstance(x, (list, set)) else 1
                )
            else:
                # Estimate from liquidity characteristics
                # More liquid loans attract more dealers
                if 'trading_volume_30d' in result.columns and 'bid_ask_spread' in result.columns:
                    volume_score = result['trading_volume_30d'].rank(pct=True)
                    spread_score = 1 - result['bid_ask_spread'].rank(pct=True)
                    liquidity_score = (volume_score + spread_score) / 2
                    # Map to 3-25 dealers based on liquidity
                    result['dealer_quote_count'] = np.round(
                        3 + liquidity_score * 22
                    ).astype('Int64')  # Nullable integer type to handle NaN
                else:
                    result['dealer_quote_count'] = np.nan

        # Calculate dealer coverage
        result['dealer_coverage'] = np.where(
            result['dealer_quote_count'].notna(),
            result['dealer_quote_count'] / self.max_dealers,
            np.nan
        )

        return result

    def calculate_clo_ownership(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CLO ownership and concentration features.

        Features calculated:
        - clo_ownership_pct: Percentage of loan held by CLOs
        - ownership_concentration: Herfindahl-Hirschman Index of top holders

        Args:
            df: DataFrame containing:
                - clo_ownership_pct (optional)
                - holder_shares (optional, list of ownership percentages)

        Returns:
            DataFrame with CLO ownership features added.
        """
        result = df.copy()

        # Calculate CLO ownership percentage
        if 'clo_ownership_pct' not in result.columns:
            if 'holder_shares' in result.columns and 'holder_types' in result.columns:
                # Calculate from holder data
                def calc_clo_pct(row):
                    if not isinstance(row['holder_shares'], list):
                        return np.nan
                    clo_shares = [
                        share for share, htype in zip(
                            row['holder_shares'], row['holder_types']
                        )
                        if htype == 'CLO'
                    ]
                    return sum(clo_shares)
                result['clo_ownership_pct'] = result.apply(calc_clo_pct, axis=1)
            else:
                # Estimate: typical CLO ownership is 50-70% for leveraged loans
                # Higher for larger, more liquid loans
                if 'facility_size' in result.columns:
                    size_factor = result['facility_size'].rank(pct=True)
                    # Map to 40-75% CLO ownership
                    result['clo_ownership_pct'] = 40 + size_factor * 35
                else:
                    result['clo_ownership_pct'] = np.nan

        # Calculate ownership concentration (HHI)
        if 'ownership_concentration' not in result.columns:
            if 'holder_shares' in result.columns:
                def calc_hhi(shares):
                    if not isinstance(shares, list) or len(shares) == 0:
                        return np.nan
                    # HHI = sum of squared market shares (0-10000 scale)
                    return sum((s ** 2) for s in shares)
                result['ownership_concentration'] = result['holder_shares'].apply(calc_hhi)
            else:
                # Estimate HHI from facility characteristics
                # Smaller loans tend to have more concentrated ownership
                if 'facility_size' in result.columns:
                    size_normalized = result['facility_size'].rank(pct=True)
                    # Larger facilities -> more dispersed ownership (lower HHI)
                    # HHI range: 500 (very dispersed) to 3000 (concentrated)
                    result['ownership_concentration'] = 3000 - size_normalized * 2500
                else:
                    result['ownership_concentration'] = np.nan

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all liquidity feature transformations.

        This method orchestrates the complete feature engineering pipeline,
        applying all individual feature calculation methods in sequence.

        Args:
            df: Input DataFrame with raw loan data.

        Returns:
            DataFrame with all liquidity features added.
        """
        result = df.copy()

        # Apply all feature calculations in sequence
        result = self.calculate_volume_features(result)
        result = self.calculate_bid_ask_features(result)
        result = self.calculate_trading_recency(result)
        result = self.calculate_dealer_metrics(result)
        result = self.calculate_clo_ownership(result)

        return result

    def get_feature_names(self) -> list:
        """
        Get list of all feature names generated by this engine.

        Returns:
            List of feature column names.
        """
        return list(self.FEATURE_SPECS.keys())

    def get_feature_descriptions(self) -> dict:
        """
        Get descriptions of all features.

        Returns:
            Dictionary mapping feature names to descriptions.
        """
        return self.FEATURE_SPECS.copy()


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan data for demonstration purposes.

    Creates realistic leveraged loan data with trading characteristics
    suitable for testing the LiquidityFeatureEngine.

    Args:
        n_samples: Number of loan samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic loan data.
    """
    rng = np.random.default_rng(seed)

    # Generate loan IDs
    loan_ids = [f"LOAN_{i:05d}" for i in range(1, n_samples + 1)]

    # Generate facility sizes (log-normal, $100M - $3B range)
    facility_sizes = rng.lognormal(mean=np.log(400), sigma=0.7, size=n_samples)
    facility_sizes = np.clip(facility_sizes, 100, 3000)
    facility_sizes = np.round(facility_sizes, 0)

    # Generate trading volumes correlated with facility size
    volume_pct = rng.uniform(0.005, 0.05, size=n_samples)
    size_factor = np.log(facility_sizes / 100) / np.log(30)
    volume_pct = volume_pct * (1 + size_factor * 0.5)
    trading_volumes = facility_sizes * volume_pct
    trading_volumes = np.round(trading_volumes, 2)

    # Generate bid-ask spreads inversely correlated with volume
    log_vol = np.log(trading_volumes + 1)
    vol_normalized = (log_vol - log_vol.min()) / (log_vol.max() - log_vol.min() + 1e-10)
    base_spread = 200 - 170 * vol_normalized
    noise = rng.normal(0, 15, size=n_samples)
    bid_ask_spreads = np.clip(base_spread + noise, 15, 250)
    bid_ask_spreads = np.round(bid_ask_spreads, 0)

    # Generate industry sectors
    sectors = [
        'Technology', 'Healthcare', 'Consumer', 'Industrials',
        'Energy', 'Financials', 'Telecom', 'Utilities'
    ]
    weights = [0.18, 0.16, 0.12, 0.12, 0.10, 0.12, 0.10, 0.10]
    industries = rng.choice(sectors, size=n_samples, p=weights)

    # Create DataFrame
    df = pd.DataFrame({
        'loan_id': loan_ids,
        'facility_size': facility_sizes,
        'trading_volume_30d': trading_volumes,
        'bid_ask_spread': bid_ask_spreads,
        'industry_sector': industries,
    })

    return df


def print_feature_summary(df: pd.DataFrame, features: list) -> None:
    """
    Print summary statistics for calculated features.

    Args:
        df: DataFrame containing calculated features.
        features: List of feature column names to summarize.
    """
    print("=" * 70)
    print("LIQUIDITY FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"\nTotal loans processed: {len(df):,}")

    for feature in features:
        if feature in df.columns:
            values = df[feature].dropna()
            if len(values) > 0:
                print(f"\n--- {feature} ---")
                print(f"  Count:  {len(values):,}")
                print(f"  Min:    {values.min():.4f}")
                print(f"  Max:    {values.max():.4f}")
                print(f"  Mean:   {values.mean():.4f}")
                print(f"  Median: {values.median():.4f}")
                print(f"  Std:    {values.std():.4f}")
            else:
                print(f"\n--- {feature} ---")
                print("  No data available")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Demonstrate usage with synthetic data
    print("Generating synthetic loan data...")
    loan_data = generate_synthetic_data(n_samples=1000, seed=42)
    print(f"Generated {len(loan_data):,} loan records")

    print("\nInitializing LiquidityFeatureEngine...")
    engine = LiquidityFeatureEngine(max_dealers=50)

    print("\nApplying feature transformations...")
    enriched_data = engine.transform(loan_data)

    # Get feature names for summary
    feature_names = engine.get_feature_names()

    # Print summary statistics
    print_feature_summary(enriched_data, feature_names)

    # Show sample of enriched data
    print("\nSample of enriched data (first 5 rows):")
    print("-" * 70)
    display_cols = ['loan_id', 'facility_size'] + [
        f for f in feature_names if f in enriched_data.columns
    ]
    print(enriched_data[display_cols].head().to_string())

    # Show feature descriptions
    print("\n" + "=" * 70)
    print("FEATURE DESCRIPTIONS")
    print("=" * 70)
    for name, desc in engine.get_feature_descriptions().items():
        print(f"  {name:30s}: {desc}")
    print("=" * 70)
