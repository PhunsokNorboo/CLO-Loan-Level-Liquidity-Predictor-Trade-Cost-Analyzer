"""
Synthetic Loan Data Generator.

Generates realistic synthetic leveraged loan data based on real market characteristics.
Designed to produce training data for liquidity prediction models when real data
is unavailable or insufficient.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class SyntheticLoanGenerator:
    """
    Generates synthetic leveraged loan data with realistic market characteristics.

    The generator creates correlated features that mimic real-world relationships:
    - Credit ratings follow typical leveraged loan distributions
    - Spreads correlate with credit quality
    - Trading volume correlates with facility size
    - Bid-ask spreads inversely correlate with trading volume
    - Liquidity tiers derive from trading characteristics
    """

    # Rating distribution based on typical leveraged loan market
    RATING_DISTRIBUTION = {
        'BB+': 0.05,
        'BB': 0.10,
        'BB-': 0.15,
        'B+': 0.25,
        'B': 0.25,
        'B-': 0.12,
        'CCC+': 0.05,
        'CCC': 0.03,
    }

    # Base spread by rating (basis points over SOFR)
    RATING_SPREAD_MAP = {
        'BB+': 200,
        'BB': 250,
        'BB-': 300,
        'B+': 350,
        'B': 425,
        'B-': 500,
        'CCC+': 600,
        'CCC': 750,
    }

    INDUSTRY_SECTORS = [
        'Technology',
        'Healthcare',
        'Consumer',
        'Industrials',
        'Energy',
        'Financials',
        'Telecom',
        'Utilities',
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize the generator with a random seed for reproducibility.

        Args:
            seed: Random seed for numpy random number generation.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_loans(self, n_loans: int = 5000) -> pd.DataFrame:
        """
        Generate a DataFrame of synthetic loan data.

        Args:
            n_loans: Number of loans to generate.

        Returns:
            DataFrame with all loan features and derived liquidity tiers.
        """
        # Generate base features
        loan_ids = [f"LOAN_{i:05d}" for i in range(1, n_loans + 1)]
        facility_sizes = self._generate_facility_sizes(n_loans)
        ratings = self._generate_ratings(n_loans)
        spreads = self._generate_spreads(ratings)
        maturities = self._generate_maturities(n_loans)
        industries = self._generate_industries(n_loans)
        volumes = self._generate_trading_volumes(n_loans, facility_sizes)
        bid_asks = self._generate_bid_ask_spreads(n_loans, volumes)
        covenant_lite = self._generate_covenant_lite(n_loans)

        # Build the DataFrame
        df = pd.DataFrame({
            'loan_id': loan_ids,
            'facility_size': facility_sizes,
            'credit_rating': ratings,
            'current_spread': spreads,
            'time_to_maturity': maturities,
            'industry_sector': industries,
            'trading_volume_30d': volumes,
            'bid_ask_spread': bid_asks,
            'covenant_lite': covenant_lite,
        })

        # Assign liquidity tiers based on trading characteristics
        df = self._assign_liquidity_tiers(df)

        return df

    def _generate_facility_sizes(self, n: int) -> np.ndarray:
        """
        Generate facility sizes following a log-normal distribution.

        Real leveraged loan facilities typically range from $100M to $3B,
        with most facilities in the $200M-$800M range.

        Args:
            n: Number of values to generate.

        Returns:
            Array of facility sizes in millions.
        """
        # Log-normal parameters calibrated to produce $100M-$3B range
        # Mean around $400M, with long tail toward larger deals
        mu = np.log(400)  # Log of median ~400M
        sigma = 0.7  # Controls spread

        sizes = self.rng.lognormal(mean=mu, sigma=sigma, size=n)

        # Clip to realistic range
        sizes = np.clip(sizes, 100, 3000)

        # Round to nearest million
        sizes = np.round(sizes, 0)

        return sizes

    def _generate_ratings(self, n: int) -> np.ndarray:
        """
        Generate credit ratings following market distribution.

        Leveraged loans are predominantly rated B+/B, with smaller
        proportions at BB tier and CCC tier.

        Args:
            n: Number of values to generate.

        Returns:
            Array of credit rating strings.
        """
        ratings = list(self.RATING_DISTRIBUTION.keys())
        probabilities = list(self.RATING_DISTRIBUTION.values())

        return self.rng.choice(ratings, size=n, p=probabilities)

    def _generate_spreads(self, ratings: np.ndarray) -> np.ndarray:
        """
        Generate spreads correlated with credit ratings.

        Higher-risk ratings have higher base spreads, with random
        variation to simulate market pricing differences.

        Args:
            ratings: Array of credit ratings.

        Returns:
            Array of spreads in basis points.
        """
        n = len(ratings)
        spreads = np.zeros(n)

        for i, rating in enumerate(ratings):
            base_spread = self.RATING_SPREAD_MAP[rating]
            # Add noise: +/- 15% variation
            noise = self.rng.normal(0, base_spread * 0.15)
            spreads[i] = base_spread + noise

        # Ensure positive spreads and round
        spreads = np.maximum(spreads, 100)
        spreads = np.round(spreads, 0)

        return spreads

    def _generate_maturities(self, n: int) -> np.ndarray:
        """
        Generate time to maturity values.

        Leveraged loans typically have 5-7 year original terms,
        so outstanding loans have 2-7 years remaining.

        Args:
            n: Number of values to generate.

        Returns:
            Array of maturities in years.
        """
        maturities = self.rng.uniform(2, 7, size=n)
        return np.round(maturities, 1)

    def _generate_industries(self, n: int) -> np.ndarray:
        """
        Generate industry sector assignments.

        Roughly equal distribution across sectors, with slight
        overweight to Technology and Healthcare.

        Args:
            n: Number of values to generate.

        Returns:
            Array of industry sector strings.
        """
        # Slight overweight to Tech and Healthcare
        weights = [0.18, 0.16, 0.12, 0.12, 0.10, 0.12, 0.10, 0.10]
        return self.rng.choice(self.INDUSTRY_SECTORS, size=n, p=weights)

    def _generate_trading_volumes(
        self, n: int, facility_sizes: np.ndarray
    ) -> np.ndarray:
        """
        Generate 30-day trading volumes correlated with facility size.

        Larger facilities tend to have higher trading volumes due to
        greater investor interest and index inclusion.

        Args:
            n: Number of values to generate.
            facility_sizes: Array of facility sizes for correlation.

        Returns:
            Array of trading volumes in millions.
        """
        # Base volume as percentage of facility size (0.5% to 5%)
        volume_pct = self.rng.uniform(0.005, 0.05, size=n)

        # Larger facilities get volume boost
        size_factor = np.log(facility_sizes / 100) / np.log(30)  # Normalized
        volume_pct = volume_pct * (1 + size_factor * 0.5)

        volumes = facility_sizes * volume_pct

        # Add some noise and ensure minimum volume
        noise = self.rng.normal(1, 0.2, size=n)
        volumes = volumes * np.maximum(noise, 0.5)
        volumes = np.maximum(volumes, 0.5)  # Minimum 0.5M

        return np.round(volumes, 2)

    def _generate_bid_ask_spreads(
        self, n: int, trading_volumes: np.ndarray
    ) -> np.ndarray:
        """
        Generate bid-ask spreads inversely correlated with trading volume.

        Higher trading volume indicates better liquidity, which
        translates to tighter bid-ask spreads.

        Args:
            n: Number of values to generate.
            trading_volumes: Array of trading volumes for correlation.

        Returns:
            Array of bid-ask spreads in basis points.
        """
        # Base bid-ask inversely related to volume
        # High volume (>20M) -> tight spread (20-40 bps)
        # Low volume (<2M) -> wide spread (100-200 bps)

        # Normalize volume to 0-1 scale (log transform for better spread)
        log_vol = np.log(trading_volumes + 1)
        vol_normalized = (log_vol - log_vol.min()) / (log_vol.max() - log_vol.min())

        # Inverse relationship: high volume -> low spread
        base_spread = 200 - 170 * vol_normalized  # Range: 30-200 bps

        # Add noise
        noise = self.rng.normal(0, 15, size=n)
        bid_asks = base_spread + noise

        # Ensure reasonable range
        bid_asks = np.clip(bid_asks, 15, 250)

        return np.round(bid_asks, 0)

    def _generate_covenant_lite(self, n: int) -> np.ndarray:
        """
        Generate covenant-lite flags.

        Approximately 70% of leveraged loans are covenant-lite,
        reflecting current market structure.

        Args:
            n: Number of values to generate.

        Returns:
            Array of boolean values.
        """
        return self.rng.choice([True, False], size=n, p=[0.70, 0.30])

    def _assign_liquidity_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign liquidity tiers based on bid-ask spread and trading volume.

        Tier definitions:
        - Tier 1: Most liquid (tight spreads, high volume)
        - Tier 2: Liquid
        - Tier 3: Moderate liquidity
        - Tier 4: Less liquid
        - Tier 5: Illiquid

        Args:
            df: DataFrame with bid_ask_spread and trading_volume_30d columns.

        Returns:
            DataFrame with liquidity_tier column added.
        """
        conditions = [
            (df['bid_ask_spread'] <= 30) & (df['trading_volume_30d'] >= 20),
            (df['bid_ask_spread'] <= 50) & (df['trading_volume_30d'] >= 10),
            (df['bid_ask_spread'] <= 100) & (df['trading_volume_30d'] >= 5),
            (df['bid_ask_spread'] <= 150),
        ]
        choices = [1, 2, 3, 4]

        df['liquidity_tier'] = np.select(conditions, choices, default=5)

        return df


def print_distribution_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for the generated loan data."""
    print("=" * 60)
    print("SYNTHETIC LOAN DATA GENERATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal loans generated: {len(df):,}")

    print("\n--- Facility Size (millions) ---")
    print(f"  Min:    ${df['facility_size'].min():,.0f}M")
    print(f"  Max:    ${df['facility_size'].max():,.0f}M")
    print(f"  Mean:   ${df['facility_size'].mean():,.0f}M")
    print(f"  Median: ${df['facility_size'].median():,.0f}M")

    print("\n--- Credit Rating Distribution ---")
    rating_counts = df['credit_rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        pct = count / len(df) * 100
        print(f"  {rating:5s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- Current Spread (bps) ---")
    print(f"  Min:    {df['current_spread'].min():.0f}")
    print(f"  Max:    {df['current_spread'].max():.0f}")
    print(f"  Mean:   {df['current_spread'].mean():.0f}")
    print(f"  Median: {df['current_spread'].median():.0f}")

    print("\n--- Time to Maturity (years) ---")
    print(f"  Min:    {df['time_to_maturity'].min():.1f}")
    print(f"  Max:    {df['time_to_maturity'].max():.1f}")
    print(f"  Mean:   {df['time_to_maturity'].mean():.1f}")

    print("\n--- Industry Sector Distribution ---")
    sector_counts = df['industry_sector'].value_counts()
    for sector, count in sector_counts.items():
        pct = count / len(df) * 100
        print(f"  {sector:12s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- Trading Volume 30d (millions) ---")
    print(f"  Min:    ${df['trading_volume_30d'].min():.2f}M")
    print(f"  Max:    ${df['trading_volume_30d'].max():.2f}M")
    print(f"  Mean:   ${df['trading_volume_30d'].mean():.2f}M")

    print("\n--- Bid-Ask Spread (bps) ---")
    print(f"  Min:    {df['bid_ask_spread'].min():.0f}")
    print(f"  Max:    {df['bid_ask_spread'].max():.0f}")
    print(f"  Mean:   {df['bid_ask_spread'].mean():.0f}")

    print("\n--- Covenant Lite ---")
    cov_lite_pct = df['covenant_lite'].mean() * 100
    print(f"  Covenant Lite: {cov_lite_pct:.1f}%")

    print("\n--- Liquidity Tier Distribution ---")
    tier_counts = df['liquidity_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        print(f"  Tier {tier}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Generate synthetic loan data
    generator = SyntheticLoanGenerator(seed=42)
    loans_df = generator.generate_loans(n_loans=5000)

    # Determine output path relative to this file
    script_dir = Path(__file__).parent.parent.parent  # Go up to project root
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "synthetic_loans.csv"

    # Save to CSV
    loans_df.to_csv(output_path, index=False)
    print(f"Saved {len(loans_df):,} loans to {output_path}")

    # Print distribution statistics
    print_distribution_stats(loans_df)
