"""
Data module for loan liquidity predictor.

Handles data acquisition from various sources:
- Kaggle datasets (Lending Club, Freddie Mac)
- FRED economic indicators
- Yahoo Finance market data
- SEC EDGAR N-PORT filings (CLO holdings)
- Web scraping for additional data
"""

from .fred_fetcher import FREDFetcher, FREDFetcherError
from .kaggle_loader import KaggleLoader
from .yfinance_fetcher import YFinanceFetcher
from .edgar_parser import EDGARParser, Filing, Position, OwnershipMetrics

__all__ = [
    'FREDFetcher',
    'FREDFetcherError',
    'KaggleLoader',
    'YFinanceFetcher',
    'EDGARParser',
    'Filing',
    'Position',
    'OwnershipMetrics',
]
