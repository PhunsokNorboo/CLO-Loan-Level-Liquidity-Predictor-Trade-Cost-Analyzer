"""
FRED API fetcher module for economic indicators.

This module provides a FREDFetcher class to retrieve economic indicators
from the Federal Reserve Economic Data (FRED) API, including:
- VIX volatility index
- Federal funds rate
- High yield spreads
- Investment grade spreads
- Yield curve slope (10Y-2Y)

The module includes rate limiting, caching, and error handling for
robust data retrieval.
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FREDFetcherError(Exception):
    """Custom exception for FRED fetcher errors."""
    pass


class FREDFetcher:
    """
    A class to fetch economic indicators from the FRED API.

    This class provides methods to retrieve various economic indicators
    commonly used in loan liquidity analysis, with built-in caching
    and rate limiting to comply with FRED API guidelines.

    Attributes:
        api_key (str): FRED API key for authentication.
        fred (Fred): The fredapi client instance.
        cache_dir (Path): Directory for caching fetched data.
        rate_limit_delay (float): Minimum seconds between API requests.
        cache_expiry_hours (int): Hours before cached data expires.

    Example:
        >>> fetcher = FREDFetcher()
        >>> vix_data = fetcher.fetch_vix('2020-01-01', '2023-12-31')
        >>> all_data = fetcher.fetch_all_indicators('2020-01-01', '2023-12-31')
    """

    # FRED series identifiers
    SERIES_VIX = 'VIXCLS'
    SERIES_FED_FUNDS = 'FEDFUNDS'
    SERIES_HY_SPREAD = 'BAMLH0A0HYM2'
    SERIES_IG_SPREAD = 'BAMLC0A0CM'
    SERIES_YIELD_CURVE = 'T10Y2Y'

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        rate_limit_delay: float = 0.5,
        cache_expiry_hours: int = 24
    ):
        """
        Initialize the FREDFetcher with API credentials and settings.

        Args:
            api_key: FRED API key. If not provided, reads from FRED_API_KEY
                environment variable.
            cache_dir: Directory for caching data. Defaults to
                'data/cache/fred/' relative to project root.
            rate_limit_delay: Minimum seconds between API requests.
                FRED allows 120 requests per minute for registered users.
            cache_expiry_hours: Number of hours before cached data is
                considered stale and re-fetched.

        Raises:
            FREDFetcherError: If no API key is available.
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('FRED_API_KEY')

        if not self.api_key:
            raise FREDFetcherError(
                "FRED API key not provided. Set FRED_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # Initialize FRED client
        self.fred = Fred(api_key=self.api_key)

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Default to project's data/cache/fred directory
            project_root = Path(__file__).parent.parent.parent
            self.cache_dir = project_root / 'data' / 'cache' / 'fred'

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting and cache settings
        self.rate_limit_delay = rate_limit_delay
        self.cache_expiry_hours = cache_expiry_hours
        self._last_request_time = 0.0

        logger.info(f"FREDFetcher initialized with cache at {self.cache_dir}")

    def _rate_limit(self) -> None:
        """
        Enforce rate limiting between API requests.

        Ensures minimum delay between consecutive API calls to avoid
        exceeding FRED's rate limits.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_cache_path(self, series_id: str, start_date: str, end_date: str) -> Path:
        """
        Generate a cache file path for the given parameters.

        Args:
            series_id: The FRED series identifier.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            Path object for the cache file.
        """
        # Create a hash of the parameters for unique filename
        params_str = f"{series_id}_{start_date}_{end_date}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        filename = f"{series_id}_{params_hash}.parquet"
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if a cache file exists and is not expired.

        Args:
            cache_path: Path to the cache file.

        Returns:
            True if cache is valid, False otherwise.
        """
        if not cache_path.exists():
            return False

        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)

        return mtime > expiry_time

    def _read_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """
        Read data from cache file.

        Args:
            cache_path: Path to the cache file.

        Returns:
            DataFrame if cache read succeeds, None otherwise.
        """
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded data from cache: {cache_path.name}")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_path}: {e}")
            return None

    def _write_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """
        Write data to cache file.

        Args:
            df: DataFrame to cache.
            cache_path: Path for the cache file.
        """
        try:
            df.to_parquet(cache_path)
            logger.info(f"Cached data to: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to write cache {cache_path}: {e}")

    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        column_name: str
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series with caching and error handling.

        Args:
            series_id: The FRED series identifier (e.g., 'VIXCLS').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            column_name: Name for the data column in output DataFrame.

        Returns:
            DataFrame with 'date' and column_name columns.

        Raises:
            FREDFetcherError: If the API request fails.
        """
        # Check cache first
        cache_path = self._get_cache_path(series_id, start_date, end_date)
        if self._is_cache_valid(cache_path):
            cached_df = self._read_cache(cache_path)
            if cached_df is not None:
                return cached_df

        # Apply rate limiting
        self._rate_limit()

        try:
            logger.info(f"Fetching {series_id} from FRED API...")
            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            # Convert to DataFrame
            df = pd.DataFrame({
                'date': series.index,
                column_name: series.values
            })
            df['date'] = pd.to_datetime(df['date'])

            # Cache the result
            self._write_cache(df, cache_path)

            logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
            return df

        except Exception as e:
            error_msg = f"Failed to fetch series {series_id}: {str(e)}"
            logger.error(error_msg)
            raise FREDFetcherError(error_msg) from e

    def fetch_vix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch VIX volatility index data.

        The VIX (CBOE Volatility Index) measures market expectations of
        near-term volatility conveyed by S&P 500 stock index option prices.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with columns: date, vix

        Example:
            >>> fetcher = FREDFetcher()
            >>> vix = fetcher.fetch_vix('2023-01-01', '2023-12-31')
            >>> print(vix.head())
        """
        return self._fetch_series(
            self.SERIES_VIX,
            start_date,
            end_date,
            'vix'
        )

    def fetch_fed_funds_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Federal Funds Effective Rate.

        The federal funds rate is the interest rate at which depository
        institutions lend reserve balances to other depository institutions
        overnight on an uncollateralized basis.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with columns: date, fed_funds_rate

        Example:
            >>> fetcher = FREDFetcher()
            >>> ffr = fetcher.fetch_fed_funds_rate('2023-01-01', '2023-12-31')
            >>> print(ffr.head())
        """
        return self._fetch_series(
            self.SERIES_FED_FUNDS,
            start_date,
            end_date,
            'fed_funds_rate'
        )

    def fetch_hy_spread(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch High Yield corporate bond spread.

        The ICE BofA US High Yield Index Option-Adjusted Spread measures
        the spread of high yield (junk) bonds over US Treasury bonds.
        Higher spreads indicate greater perceived credit risk.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with columns: date, hy_spread

        Example:
            >>> fetcher = FREDFetcher()
            >>> hy = fetcher.fetch_hy_spread('2023-01-01', '2023-12-31')
            >>> print(hy.head())
        """
        return self._fetch_series(
            self.SERIES_HY_SPREAD,
            start_date,
            end_date,
            'hy_spread'
        )

    def fetch_ig_spread(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Investment Grade corporate bond spread.

        The ICE BofA US Corporate Index Option-Adjusted Spread measures
        the spread of investment grade corporate bonds over US Treasury bonds.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with columns: date, ig_spread

        Example:
            >>> fetcher = FREDFetcher()
            >>> ig = fetcher.fetch_ig_spread('2023-01-01', '2023-12-31')
            >>> print(ig.head())
        """
        return self._fetch_series(
            self.SERIES_IG_SPREAD,
            start_date,
            end_date,
            'ig_spread'
        )

    def fetch_yield_curve(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch yield curve slope (10-Year minus 2-Year Treasury spread).

        The 10Y-2Y Treasury spread is a key indicator of economic conditions.
        A negative (inverted) yield curve has historically preceded recessions.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with columns: date, yield_curve_slope

        Example:
            >>> fetcher = FREDFetcher()
            >>> yc = fetcher.fetch_yield_curve('2023-01-01', '2023-12-31')
            >>> print(yc.head())
        """
        return self._fetch_series(
            self.SERIES_YIELD_CURVE,
            start_date,
            end_date,
            'yield_curve_slope'
        )

    def fetch_all_indicators(
        self,
        start_date: str,
        end_date: str,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Fetch all economic indicators and combine into a single DataFrame.

        This method fetches VIX, Federal Funds Rate, High Yield Spread,
        Investment Grade Spread, and Yield Curve Slope, then merges them
        on the date column. Missing values are handled using the specified
        fill method.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            fill_method: Method to fill missing values. Options:
                - 'ffill': Forward fill (default)
                - 'bfill': Backward fill
                - 'interpolate': Linear interpolation
                - None: Leave missing values as NaN

        Returns:
            DataFrame with columns:
                - date: The observation date
                - vix: VIX volatility index
                - fed_funds_rate: Federal funds effective rate
                - hy_spread: High yield bond spread
                - ig_spread: Investment grade bond spread
                - yield_curve_slope: 10Y-2Y Treasury spread

        Example:
            >>> fetcher = FREDFetcher()
            >>> df = fetcher.fetch_all_indicators('2020-01-01', '2023-12-31')
            >>> print(df.head())
            >>> print(df.describe())
        """
        logger.info(f"Fetching all indicators from {start_date} to {end_date}")

        # Fetch each series
        dfs = []
        fetch_methods = [
            ('VIX', self.fetch_vix),
            ('Fed Funds Rate', self.fetch_fed_funds_rate),
            ('HY Spread', self.fetch_hy_spread),
            ('IG Spread', self.fetch_ig_spread),
            ('Yield Curve', self.fetch_yield_curve),
        ]

        for name, method in fetch_methods:
            try:
                df = method(start_date, end_date)
                dfs.append(df)
                logger.info(f"Fetched {name}: {len(df)} observations")
            except FREDFetcherError as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                continue

        if not dfs:
            raise FREDFetcherError("Failed to fetch any indicators")

        # Merge all DataFrames on date
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='date', how='outer')

        # Sort by date
        result = result.sort_values('date').reset_index(drop=True)

        # Handle missing values
        if fill_method == 'ffill':
            result = result.ffill()
        elif fill_method == 'bfill':
            result = result.bfill()
        elif fill_method == 'interpolate':
            result = result.interpolate(method='linear')
        # If None, leave NaN values as-is

        logger.info(f"Combined dataset: {len(result)} rows, {len(result.columns)} columns")

        return result

    def clear_cache(self) -> int:
        """
        Clear all cached data files.

        Returns:
            Number of cache files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob('*.parquet'):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cache files")
        return count


if __name__ == "__main__":
    """
    Demonstration of FREDFetcher usage.

    This example shows how to:
    1. Initialize the fetcher with API credentials
    2. Fetch individual indicators
    3. Fetch all indicators combined
    4. Display basic statistics
    """
    print("=" * 60)
    print("FRED Economic Indicators Fetcher - Demo")
    print("=" * 60)

    # Define date range for demonstration
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    try:
        # Initialize the fetcher
        print("\n1. Initializing FREDFetcher...")
        fetcher = FREDFetcher()
        print("   Fetcher initialized successfully!")

        # Fetch individual indicator (VIX as example)
        print(f"\n2. Fetching VIX data ({start_date} to {end_date})...")
        vix_df = fetcher.fetch_vix(start_date, end_date)
        print(f"   Retrieved {len(vix_df)} observations")
        print("\n   VIX Sample (first 5 rows):")
        print(vix_df.head().to_string(index=False))

        # Fetch all indicators
        print(f"\n3. Fetching all indicators ({start_date} to {end_date})...")
        all_df = fetcher.fetch_all_indicators(start_date, end_date)

        print(f"\n   Combined dataset shape: {all_df.shape}")
        print("\n   Sample data (first 5 rows):")
        print(all_df.head().to_string(index=False))

        # Show summary statistics
        print("\n   Summary Statistics:")
        print("-" * 60)
        stats = all_df.describe()
        numeric_cols = [c for c in all_df.columns if c != 'date']
        for col in numeric_cols:
            print(f"\n   {col}:")
            print(f"      Mean: {stats.loc['mean', col]:.4f}")
            print(f"      Std:  {stats.loc['std', col]:.4f}")
            print(f"      Min:  {stats.loc['min', col]:.4f}")
            print(f"      Max:  {stats.loc['max', col]:.4f}")

        # Show data completeness
        print("\n   Data Completeness:")
        print("-" * 60)
        for col in all_df.columns:
            non_null = all_df[col].notna().sum()
            pct = 100 * non_null / len(all_df)
            print(f"   {col}: {non_null}/{len(all_df)} ({pct:.1f}%)")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except FREDFetcherError as e:
        print(f"\nError: {e}")
        print("\nTo use this module, ensure you have a valid FRED API key.")
        print("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\nThen set it as an environment variable:")
        print("  export FRED_API_KEY='your-api-key-here'")
        print("\nOr pass it directly to the FREDFetcher:")
        print("  fetcher = FREDFetcher(api_key='your-api-key-here')")
