# CLO Loan-Level Liquidity Predictor & Trade Cost Analyzer

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

A machine learning system for predicting loan-level liquidity and estimating trade execution costs in the CLO (Collateralized Loan Obligation) and leveraged loan markets.

## Overview

Liquidity in the leveraged loan market is notoriously opaque. Unlike equities or bonds, leveraged loans trade over-the-counter with limited price transparency, making it difficult for portfolio managers to assess execution costs before trading. This project addresses that challenge by building predictive models that estimate both **liquidity tiers** and **expected bid-ask spreads** at the individual loan level.

The CLO market represents over $1 trillion in assets under management, with leveraged loans as the primary underlying collateral. Accurate liquidity prediction is critical for CLO managers who must balance portfolio returns against trading costs, regulatory constraints, and redemption pressures. This tool provides actionable intelligence by combining loan-specific characteristics with real-time market conditions.

Key predictions include:
- **Liquidity Tier Classification (1-5)**: XGBoost classifier predicting how liquid a loan is relative to the market
- **Expected Bid-Ask Spread (bps)**: LightGBM regressor estimating transaction costs
- **SHAP Explainability**: Feature importance and individual prediction explanations for transparency and compliance

## Features

- **Multi-source data integration**: FRED economic indicators, SEC EDGAR filings, Yahoo Finance market data
- **Comprehensive feature engineering**: 30+ engineered features across loan, market, and liquidity dimensions
- **Dual-model architecture**: Classification for liquidity tiers, regression for spread prediction
- **SHAP-based explainability**: Global feature importance and local prediction explanations
- **Interactive Streamlit dashboard**: Real-time predictions with visualization
- **Modular design**: Clean separation of data, features, models, and explainability components
- **Caching and rate limiting**: Efficient API usage with local caching for all data sources

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/loan-liquidity-predictor.git
cd loan-liquidity-predictor

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys (FRED_API_KEY required for economic data)
```

### API Keys

- **FRED API Key** (required): Get a free key at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

## Quick Start

```bash
# Generate synthetic training data
python -m src.data.data_generator

# Run feature engineering demo
python -m src.features.loan_features

# Launch the Streamlit app
streamlit run streamlit_app.py
```

## Data Sources

| Source | Data Type | Usage | Rate Limit |
|--------|-----------|-------|------------|
| **FRED API** | Economic indicators | VIX, HY/IG spreads, Fed funds, yield curve | 120 req/min |
| **SEC EDGAR** | N-PORT filings | CLO ownership concentration, fund holdings | 10 req/sec |
| **Yahoo Finance** | Market data | VIX, S&P 500, credit ETFs (HYG, LQD), sector ETFs | Best effort |
| **Synthetic** | Loan data | Training and demonstration | N/A |

### FRED Series Used
- `VIXCLS`: VIX volatility index
- `FEDFUNDS`: Federal funds effective rate
- `BAMLH0A0HYM2`: ICE BofA High Yield spread
- `BAMLC0A0CM`: ICE BofA Investment Grade spread
- `T10Y2Y`: 10-Year minus 2-Year Treasury spread

## Model Architecture

### Liquidity Tier Classifier (XGBoost)

Predicts loan liquidity tier (1-5) based on trading characteristics:

| Tier | Description | Typical Bid-Ask | Trading Volume |
|------|-------------|-----------------|----------------|
| 1 | Most Liquid | < 30 bps | > $20M/month |
| 2 | Liquid | 30-50 bps | $10-20M/month |
| 3 | Moderate | 50-100 bps | $5-10M/month |
| 4 | Less Liquid | 100-150 bps | $1-5M/month |
| 5 | Illiquid | > 150 bps | < $1M/month |

### Trade Cost Predictor (LightGBM)

Regresses expected bid-ask spread in basis points. Useful for:
- Pre-trade cost estimation
- Execution algorithm selection
- Portfolio rebalancing optimization

### SHAP Explainability

Both models include SHAP (SHapley Additive exPlanations) integration for:
- **Global importance**: Which features drive predictions across the portfolio
- **Local explanations**: Why a specific loan received its prediction
- **Compliance documentation**: Audit trail for trading decisions

## Feature Engineering

### Loan-Level Features (`src/features/loan_features.py`)

| Feature | Description |
|---------|-------------|
| `facility_size_log` | Log-transformed facility size |
| `facility_size_pctl` | Percentile rank of facility size |
| `credit_rating_encoded` | Ordinal encoding (BB+ = 1 to CCC = 8) |
| `rating_*` | One-hot encoded credit ratings |
| `spread_z_score` | Z-score of spread within rating category |
| `time_to_maturity` | Years until maturity |
| `maturity_bucket` | Short/Medium/Long categorization |
| `near_maturity` | Flag for loans < 1 year to maturity |
| `sector_*` | One-hot encoded industry sectors |
| `covenant_lite` | Covenant-lite indicator |

### Market-Level Features (`src/features/market_features.py`)

| Feature | Description |
|---------|-------------|
| `vix_level` | Current VIX value |
| `vix_percentile` | Rolling 252-day VIX percentile |
| `vix_regime` | Low/Normal/High/Extreme classification |
| `hy_spread` | High yield credit spread |
| `ig_spread` | Investment grade credit spread |
| `hy_ig_gap` | HY minus IG spread (credit differentiation) |
| `fed_funds_rate` | Current Fed funds rate |
| `fed_funds_change_30d` | 30-day change in Fed funds |
| `yield_curve_slope` | 10Y-2Y Treasury spread |
| `curve_inverted` | Yield curve inversion flag |
| `market_stress` | Composite stress indicator (0-1) |

### Liquidity Features (`src/features/liquidity_features.py`)

| Feature | Description |
|---------|-------------|
| `volume_percentile` | Percentile rank of 30-day volume |
| `volume_to_size_ratio` | Turnover ratio |
| `bid_ask_percentile` | Percentile rank of spread |
| `spread_volatility` | Bid-ask spread volatility |
| `days_since_last_trade` | Trading recency |
| `trade_frequency` | Trades per week |
| `dealer_quote_count` | Number of dealers quoting |
| `dealer_coverage` | Quote count / max dealers |
| `clo_ownership_pct` | CLO ownership percentage |
| `ownership_concentration` | HHI of top holders |

## Usage Examples

### Fetching Economic Data

```python
from src.data.fred_fetcher import FREDFetcher

fetcher = FREDFetcher()
indicators = fetcher.fetch_all_indicators('2023-01-01', '2023-12-31')
print(indicators.head())
```

### Engineering Loan Features

```python
from src.features.loan_features import LoanFeatureEngine
import pandas as pd

# Load your loan data
loans = pd.read_csv('data/synthetic_loans.csv')

# Transform features
engine = LoanFeatureEngine()
features = engine.transform(loans)
print(f"Engineered {len(features.columns)} features")
```

### Parsing SEC EDGAR Filings

```python
from src.data.edgar_parser import EDGARParser

parser = EDGARParser(user_agent="YourApp research@yourdomain.com")

# Get CLO holdings for a fund
holdings = parser.get_clo_holdings_for_fund(cik="1006249", filing_count=1)
for h in holdings:
    print(f"Fund: {h['fund_name']}")
    print(f"CLO Positions: {h['clo_positions']}")
    print(f"Top 5 Concentration: {h['top_5_concentration']:.1f}%")
```

### Market Feature Engineering

```python
from src.features.market_features import MarketFeatureEngine
from src.data.fred_fetcher import FREDFetcher

# Fetch raw market data
fetcher = FREDFetcher()
raw_data = fetcher.fetch_all_indicators('2023-01-01', '2023-12-31')

# Engineer features
engine = MarketFeatureEngine()
market_features = engine.transform(raw_data)
print(f"Market stress indicator: {market_features['market_stress'].iloc[-1]:.3f}")
```

## Project Structure

```
loan-liquidity-predictor/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore patterns
├── streamlit_app.py                   # Interactive dashboard
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Shared utilities
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_generator.py          # Synthetic loan data generator
│   │   ├── fred_fetcher.py            # FRED API client
│   │   ├── yfinance_fetcher.py        # Yahoo Finance client
│   │   ├── edgar_parser.py            # SEC EDGAR N-PORT parser
│   │   └── kaggle_loader.py           # LendingClub data loader
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── loan_features.py           # Loan-level feature engineering
│   │   ├── market_features.py         # Market condition features
│   │   └── liquidity_features.py      # Liquidity indicator features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── liquidity_model.py         # XGBoost liquidity classifier
│   │   └── spread_model.py            # LightGBM spread predictor
│   │
│   └── explainability/
│       ├── __init__.py
│       └── shap_utils.py              # SHAP explanation utilities
│
├── notebooks/
│   ├── 01_data_collection.ipynb       # Data gathering walkthrough
│   ├── 02_exploratory_analysis.ipynb  # EDA and visualization
│   ├── 03_feature_engineering.ipynb   # Feature development
│   ├── 04_model_training.ipynb        # Model training and tuning
│   └── 05_model_evaluation.ipynb      # Performance evaluation
│
├── data/
│   ├── synthetic_loans.csv            # Generated training data
│   └── cache/                         # Cached API responses
│       ├── fred/
│       ├── yfinance/
│       └── edgar/
│
├── models/                            # Saved model artifacts
│   ├── liquidity_classifier.joblib
│   └── spread_regressor.joblib
│
└── reports/                           # Generated analysis reports
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure code passes linting (`flake8`, `black`)
5. Submit a pull request with a clear description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/

# Check linting
flake8 src/ tests/
```

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
