"""
Yahoo Finance data fetcher for market indicators.

Fetches market data relevant to loan liquidity prediction:
- VIX (volatility index)
- S&P 500 (broad market)
- Credit ETFs (HYG, LQD for credit spreads)
- Sector ETFs (financial, tech, healthcare, energy, industrial)

Data is cached locally to avoid redundant API calls.
"""

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.utils import get_data_dir, setup_logging

logger = logging.getLogger("loan_liquidity_predictor.yfinance")


class YFinanceFetcher:
    """
    Fetches and caches market data from Yahoo Finance.

    Provides methods to retrieve market indicators that may influence
    loan liquidity, including volatility indices, broad market indices,
    and credit-related ETFs.

    Attributes:
        cache_dir: Path to the cache directory for storing fetched data.
        delay_seconds: Delay between API requests to avoid rate limiting.
    """

    # Standard tickers for different asset classes
    VIX_TICKER = "^VIX"
    SP500_TICKER = "^GSPC"
    CREDIT_ETFS = ["HYG", "LQD"]  # High Yield, Investment Grade
    SECTOR_ETFS = [
        "XLF",  # Financial
        "XLK",  # Technology
        "XLV",  # Healthcare
        "XLE",  # Energy
        "XLI",  # Industrial
        "XLU",  # Utilities
        "XLP",  # Consumer Staples
        "XLY",  # Consumer Discretionary
        "XLB",  # Materials
        "XLRE", # Real Estate
    ]

    def __init__(self, delay_seconds: float = 0.5):
        """
        Initialize the Yahoo Finance fetcher.

        Args:
            delay_seconds: Time to wait between API requests to avoid
                rate limiting. Default is 0.5 seconds.
        """
        self.cache_dir = get_data_dir("cache/yfinance")
        self.delay_seconds = delay_seconds
        logger.info(f"YFinanceFetcher initialized. Cache dir: {self.cache_dir}")

    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> Path:
        """
        Generate a cache file path based on ticker and date range.

        Args:
            ticker: The stock/index ticker symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            Path to the cache file.
        """
        # Sanitize ticker for filesystem (remove ^ and other special chars)
        safe_ticker = ticker.replace("^", "").replace("/", "_")
        filename = f"{safe_ticker}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from cache if it exists.

        Args:
            cache_path: Path to the cache file.

        Returns:
            DataFrame if cache exists and is valid, None otherwise.
        """
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded from cache: {cache_path}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """
        Save data to cache.

        Args:
            df: DataFrame to save.
            cache_path: Path to save the cache file.
        """
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=True)
            logger.debug(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def _fetch_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for a single ticker with caching.

        Args:
            ticker: The ticker symbol to fetch.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with OHLCV data for the ticker.
        """
        cache_path = self._get_cache_path(ticker, start_date, end_date)

        # Try loading from cache
        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data

        # Fetch from Yahoo Finance
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Standardize index name
            df.index.name = "date"
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # Save to cache
            if use_cache:
                self._save_to_cache(df, cache_path)

            # Rate limiting delay
            time.sleep(self.delay_seconds)

            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def _fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data using forward fill then backward fill.

        Args:
            df: DataFrame with potential missing values.

        Returns:
            DataFrame with missing values filled.
        """
        if df.empty:
            return df
        df = df.ffill()
        df = df.bfill()
        return df

    def fetch_vix(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch VIX (Volatility Index) data.

        The VIX measures market expectation of near-term volatility
        based on S&P 500 index options.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with VIX data, columns: Open, High, Low, Close, Volume.
        """
        df = self._fetch_ticker(self.VIX_TICKER, start_date, end_date, use_cache)
        return self._fill_missing_data(df)

    def fetch_sp500(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch S&P 500 index data.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with S&P 500 data, columns: Open, High, Low, Close, Volume.
        """
        df = self._fetch_ticker(self.SP500_TICKER, start_date, end_date, use_cache)
        return self._fill_missing_data(df)

    def fetch_credit_etfs(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch credit ETF data (HYG and LQD).

        HYG tracks high-yield corporate bonds.
        LQD tracks investment-grade corporate bonds.
        The spread between them indicates credit market stress.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with credit ETF close prices, indexed by date.
        """
        dfs = {}
        for ticker in self.CREDIT_ETFS:
            df = self._fetch_ticker(ticker, start_date, end_date, use_cache)
            if not df.empty:
                dfs[ticker.lower()] = df["Close"]

        if not dfs:
            return pd.DataFrame()

        result = pd.DataFrame(dfs)
        result.index.name = "date"
        return self._fill_missing_data(result)

    def fetch_sector_etfs(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch sector ETF data.

        Includes: XLF (Financial), XLK (Tech), XLV (Healthcare),
        XLE (Energy), XLI (Industrial), and more.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with sector ETF close prices, indexed by date.
        """
        dfs = {}
        for ticker in self.SECTOR_ETFS:
            df = self._fetch_ticker(ticker, start_date, end_date, use_cache)
            if not df.empty:
                dfs[ticker.lower()] = df["Close"]

        if not dfs:
            return pd.DataFrame()

        result = pd.DataFrame(dfs)
        result.index.name = "date"
        return self._fill_missing_data(result)

    def calculate_market_stress_indicator(
        self,
        df: pd.DataFrame,
        vix_column: str = "vix_close",
        window: int = 90
    ) -> pd.Series:
        """
        Calculate market stress indicator based on VIX.

        The stress indicator is the ratio of current VIX to its
        90-day rolling mean. Values above 1.0 indicate elevated stress.

        Args:
            df: DataFrame containing VIX data.
            vix_column: Name of the column containing VIX close prices.
            window: Rolling window size in days (default 90).

        Returns:
            Series with market stress indicator values.
        """
        if vix_column not in df.columns:
            logger.error(f"Column {vix_column} not found in DataFrame")
            return pd.Series(dtype=float)

        vix = df[vix_column]
        rolling_mean = vix.rolling(window=window, min_periods=1).mean()
        stress_indicator = vix / rolling_mean

        return stress_indicator

    def fetch_all_market_data(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch and combine all market data into a single DataFrame.

        Combines VIX, S&P 500, and credit ETF data. Also calculates
        derived metrics like the high-yield/investment-grade spread
        and the market stress indicator.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with columns:
                - date (index)
                - vix_close
                - sp500_close
                - hyg_close (high yield ETF)
                - lqd_close (investment grade ETF)
                - hy_ig_spread (HYG - LQD price spread proxy)
                - market_stress (VIX / 90-day rolling mean)
        """
        logger.info(f"Fetching all market data from {start_date} to {end_date}")

        # Fetch individual datasets
        vix_df = self.fetch_vix(start_date, end_date, use_cache)
        sp500_df = self.fetch_sp500(start_date, end_date, use_cache)
        credit_df = self.fetch_credit_etfs(start_date, end_date, use_cache)

        # Build combined DataFrame
        combined = pd.DataFrame()

        # Add VIX close
        if not vix_df.empty:
            combined["vix_close"] = vix_df["Close"]

        # Add S&P 500 close
        if not sp500_df.empty:
            combined["sp500_close"] = sp500_df["Close"]

        # Add credit ETF data
        if not credit_df.empty:
            if "hyg" in credit_df.columns:
                combined["hyg_close"] = credit_df["hyg"]
            if "lqd" in credit_df.columns:
                combined["lqd_close"] = credit_df["lqd"]

        if combined.empty:
            logger.warning("No data fetched - returning empty DataFrame")
            return pd.DataFrame()

        # Fill any missing data after joining
        combined = self._fill_missing_data(combined)

        # Calculate HY-IG spread (price-based proxy)
        # A widening spread (HYG underperforming LQD) indicates credit stress
        if "hyg_close" in combined.columns and "lqd_close" in combined.columns:
            # Normalize to percentage returns for spread calculation
            hyg_norm = combined["hyg_close"] / combined["hyg_close"].iloc[0] * 100
            lqd_norm = combined["lqd_close"] / combined["lqd_close"].iloc[0] * 100
            combined["hy_ig_spread"] = hyg_norm - lqd_norm

        # Calculate market stress indicator
        if "vix_close" in combined.columns:
            combined["market_stress"] = self.calculate_market_stress_indicator(
                combined, "vix_close"
            )

        # Ensure index is named correctly
        combined.index.name = "date"

        logger.info(f"Combined data shape: {combined.shape}")
        return combined


if __name__ == "__main__":
    """
    Demonstrate usage of the YFinanceFetcher class.
    """
    # Set up logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize fetcher
    fetcher = YFinanceFetcher(delay_seconds=0.5)

    # Define date range (last 2 years for demonstration)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = "2023-01-01"

    print("\n" + "=" * 60)
    print("Yahoo Finance Market Data Fetcher - Demo")
    print("=" * 60)

    # Fetch individual datasets
    print("\n1. Fetching VIX data...")
    vix_df = fetcher.fetch_vix(start_date, end_date)
    if not vix_df.empty:
        print(f"   VIX shape: {vix_df.shape}")
        print(f"   VIX range: {vix_df['Close'].min():.2f} - {vix_df['Close'].max():.2f}")

    print("\n2. Fetching S&P 500 data...")
    sp500_df = fetcher.fetch_sp500(start_date, end_date)
    if not sp500_df.empty:
        print(f"   S&P 500 shape: {sp500_df.shape}")
        print(f"   S&P 500 range: {sp500_df['Close'].min():.2f} - {sp500_df['Close'].max():.2f}")

    print("\n3. Fetching credit ETF data...")
    credit_df = fetcher.fetch_credit_etfs(start_date, end_date)
    if not credit_df.empty:
        print(f"   Credit ETF shape: {credit_df.shape}")
        print(f"   Columns: {list(credit_df.columns)}")

    print("\n4. Fetching sector ETF data...")
    sector_df = fetcher.fetch_sector_etfs(start_date, end_date)
    if not sector_df.empty:
        print(f"   Sector ETF shape: {sector_df.shape}")
        print(f"   Columns: {list(sector_df.columns)}")

    # Fetch combined data
    print("\n5. Fetching combined market data...")
    combined_df = fetcher.fetch_all_market_data(start_date, end_date)

    if not combined_df.empty:
        print(f"\nCombined DataFrame:")
        print(f"   Shape: {combined_df.shape}")
        print(f"   Columns: {list(combined_df.columns)}")
        print(f"   Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"\n   Sample data (last 5 rows):")
        print(combined_df.tail().to_string())

        # Summary statistics
        print(f"\n   Summary statistics:")
        print(combined_df.describe().round(2).to_string())

        # Market stress analysis
        if "market_stress" in combined_df.columns:
            stress = combined_df["market_stress"]
            print(f"\n   Market Stress Indicator:")
            print(f"      Current: {stress.iloc[-1]:.3f}")
            print(f"      Mean: {stress.mean():.3f}")
            print(f"      Max: {stress.max():.3f} on {stress.idxmax().strftime('%Y-%m-%d')}")
            print(f"      Days with stress > 1.2: {(stress > 1.2).sum()}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
