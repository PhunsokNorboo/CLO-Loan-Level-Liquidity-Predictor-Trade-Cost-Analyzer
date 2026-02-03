#!/usr/bin/env python3
"""
Executive Summary Report Generator for CLO Loan-Level Liquidity Predictor.

Generates a professional PDF report summarizing the project's capabilities,
model performance, and business relevance for Octaura interview presentation.
"""

import warnings
# Suppress fpdf2 deprecation warnings (using legacy parameter style for compatibility)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from datetime import datetime
from pathlib import Path
from fpdf import FPDF


class ExecutiveReport(FPDF):
    """Custom PDF class with professional header and footer."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Add header with title bar."""
        # Header background
        self.set_fill_color(25, 55, 95)  # Dark blue
        self.rect(0, 0, 210, 18, 'F')

        # Header text
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 6, 'CLO Loan-Level Liquidity Predictor | Executive Summary', 0, 0, 'L')

        # Reset for body
        self.set_text_color(0, 0, 0)
        self.ln(20)

    def footer(self):
        """Add footer with page number and generation info."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.set_x(10)
        self.cell(0, 10, f'Generated {datetime.now().strftime("%Y-%m-%d")} with Claude Code', 0, 0, 'L')

    def section_title(self, title: str, color: tuple = (25, 55, 95)):
        """Add a styled section title."""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(*color)
        self.cell(0, 8, title, 0, 1, 'L')
        # Underline
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title: str):
        """Add a subsection title."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text: str):
        """Add body text with proper formatting."""
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text: str, indent: int = 10):
        """Add a bullet point."""
        self.set_font('Helvetica', '', 10)
        self.set_x(indent)
        self.cell(5, 5, chr(149), 0, 0)  # Bullet character
        self.multi_cell(0, 5, text)

    def metric_box(self, label: str, value: str, target: str = None):
        """Add a metric with optional target comparison."""
        self.set_font('Helvetica', 'B', 9)
        self.cell(60, 6, label + ':', 0, 0)
        self.set_font('Helvetica', '', 9)
        if target:
            self.cell(40, 6, value, 0, 0)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, f'(Target: {target})', 0, 1)
            self.set_text_color(0, 0, 0)
        else:
            self.cell(0, 6, value, 0, 1)

    def add_table(self, headers: list, data: list, col_widths: list = None):
        """Add a simple table."""
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)

        # Header row
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', True)
        self.ln()

        # Data rows
        self.set_font('Helvetica', '', 9)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'C')
            self.ln()
        self.ln(3)


def generate_report():
    """Generate the executive summary PDF report."""
    pdf = ExecutiveReport()
    pdf.alias_nb_pages()

    # =========================================================================
    # PAGE 1: Executive Summary
    # =========================================================================
    pdf.add_page()

    # Main title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(25, 55, 95)
    pdf.cell(0, 12, 'CLO Loan-Level Liquidity Predictor', 0, 1, 'C')

    pdf.set_font('Helvetica', 'I', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, 'ML-Based Trade Cost Analysis for Leveraged Loans', 0, 1, 'C')
    pdf.ln(8)

    # Overview Section
    pdf.section_title('Overview')
    pdf.body_text(
        'Liquidity in the leveraged loan market is notoriously opaque. Unlike equities or bonds, '
        'leveraged loans trade over-the-counter with limited price transparency, creating significant '
        'challenges for trade execution and portfolio management.'
    )
    pdf.body_text(
        'This project addresses that challenge by building machine learning models that predict '
        'loan-level liquidity tiers and expected bid-ask spreads. The CLO market represents over '
        '$1 trillion in assets under management, making accurate liquidity prediction critical for '
        'CLO managers balancing portfolio returns against trading costs.'
    )
    pdf.ln(2)

    # Key Capabilities
    pdf.section_title('Key Capabilities')
    pdf.bullet_point('Liquidity Tier Classification (1-5 scale) - XGBoost classifier predicting relative loan liquidity')
    pdf.bullet_point('Bid-Ask Spread Prediction (bps) - LightGBM regressor estimating transaction costs')
    pdf.bullet_point('Feature-based Explainability - SHAP TreeExplainer for transparent, auditable predictions')
    pdf.bullet_point('Interactive Dashboard - Streamlit app for real-time predictions and visualization')
    pdf.ln(4)

    # Liquidity Tier Definitions Table
    pdf.section_title('Liquidity Tier Definitions')
    pdf.add_table(
        headers=['Tier', 'Description', 'Typical Bid-Ask', 'Trading Volume'],
        data=[
            ['1', 'Most Liquid', '< 30 bps', '> $20M/month'],
            ['2', 'Liquid', '30-50 bps', '$10-20M/month'],
            ['3', 'Moderate', '50-100 bps', '$5-10M/month'],
            ['4', 'Less Liquid', '100-150 bps', '$1-5M/month'],
            ['5', 'Illiquid', '> 150 bps', '< $1M/month'],
        ],
        col_widths=[20, 45, 55, 70]
    )

    # =========================================================================
    # PAGE 2: Model Performance
    # =========================================================================
    pdf.add_page()
    pdf.section_title('Model Performance')

    # Liquidity Tier Model
    pdf.subsection_title('Liquidity Tier Classifier (XGBoost)')
    pdf.body_text(
        'Multi-class classification model predicting loan liquidity tiers (1-5) based on '
        'loan characteristics, market conditions, and trading metrics.'
    )
    pdf.metric_box('Algorithm', 'XGBoost Gradient Boosting')
    pdf.metric_box('Validation', 'TimeSeriesSplit 5-fold CV (prevents look-ahead bias)')
    pdf.metric_box('Accuracy', '~99% on synthetic data', '>70%')
    pdf.metric_box('Output', '5-class probabilities with SHAP explanations')
    pdf.ln(4)

    # Trade Cost Model
    pdf.subsection_title('Trade Cost Predictor (LightGBM)')
    pdf.body_text(
        'Regression model estimating expected bid-ask spreads in basis points, enabling '
        'pre-trade cost analysis and execution optimization.'
    )
    pdf.metric_box('Algorithm', 'LightGBM Gradient Boosting')
    pdf.metric_box('Validation', '5-fold Cross-Validation')
    pdf.metric_box('MAE', '~12 bps', '<30 bps')
    pdf.metric_box('R-squared', '>0.85')
    pdf.metric_box('Confidence Intervals', '95% CI via bootstrap resampling')
    pdf.ln(4)

    # Top Predictive Features
    pdf.section_title('Top Predictive Features')
    pdf.add_table(
        headers=['Rank', 'Feature', 'Category', 'Description'],
        data=[
            ['1', 'trading_volume_30d', 'Liquidity', '30-day trading volume'],
            ['2', 'bid_ask_spread', 'Liquidity', 'Current bid-ask spread'],
            ['3', 'facility_size', 'Loan', 'Total facility amount'],
            ['4', 'credit_rating_encoded', 'Loan', 'Ordinal credit rating (BB+ to CCC)'],
            ['5', 'market_stress', 'Market', 'Composite stress indicator'],
            ['6', 'dealer_quote_count', 'Liquidity', 'Number of dealers quoting'],
            ['7', 'clo_ownership_pct', 'Ownership', 'CLO ownership percentage'],
            ['8', 'vix_level', 'Market', 'VIX volatility index'],
        ],
        col_widths=[15, 50, 35, 90]
    )

    # =========================================================================
    # PAGE 3: Technical Architecture & Business Relevance
    # =========================================================================
    pdf.add_page()
    pdf.section_title('Technical Architecture')

    # Data Sources
    pdf.subsection_title('Data Sources')
    pdf.add_table(
        headers=['Source', 'Data Type', 'Usage'],
        data=[
            ['FRED API', 'Economic Indicators', 'VIX, HY/IG spreads, Fed funds, yield curve'],
            ['SEC EDGAR', 'N-PORT Filings', 'CLO ownership concentration, fund holdings'],
            ['Yahoo Finance', 'Market Data', 'VIX, S&P 500, credit ETFs (HYG, LQD)'],
            ['Synthetic Generator', 'Training Data', 'Realistic loan data for model development'],
        ],
        col_widths=[45, 50, 95]
    )

    # Feature Engineering
    pdf.subsection_title('Feature Engineering Pipeline')
    pdf.body_text(
        '30+ engineered features across three dimensions:'
    )
    pdf.bullet_point('Loan Features: facility_size_log, credit_rating_encoded, spread_z_score, time_to_maturity, covenant_lite')
    pdf.bullet_point('Market Features: vix_level, vix_percentile, hy_spread, ig_spread, yield_curve_slope, market_stress')
    pdf.bullet_point('Liquidity Features: volume_percentile, bid_ask_percentile, days_since_trade, dealer_coverage, clo_ownership_pct')
    pdf.ln(3)

    # Model Architecture Diagram (text-based)
    pdf.subsection_title('Model Pipeline')
    pdf.set_font('Courier', '', 9)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(190, 5, '  Raw Data --> Feature Engineering --> Model Training --> SHAP Explainer', 1, 1, 'L', True)
    pdf.cell(190, 5, '      |              |                      |                  |', 1, 1, 'L', True)
    pdf.cell(190, 5, '  [Loan,Market,   [Loan/Market/          [XGBoost +       [Global +', 1, 1, 'L', True)
    pdf.cell(190, 5, '   Liquidity]     Liquidity Engines]     LightGBM]       Local SHAP]', 1, 1, 'L', True)
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(4)

    # Business Relevance
    pdf.section_title('Business Relevance to Octaura')
    pdf.body_text(
        'This project directly addresses core challenges in the CLO and leveraged loan markets '
        'that Octaura aims to solve:'
    )

    pdf.subsection_title('Trade Execution Quality')
    pdf.bullet_point('Pre-trade cost estimation enables optimal execution timing')
    pdf.bullet_point('Confidence intervals quantify execution risk')
    pdf.bullet_point('Liquidity tier classification guides order sizing')
    pdf.ln(2)

    pdf.subsection_title('Mark-to-Market Pricing')
    pdf.bullet_point('Spread predictions support fair value estimation for illiquid positions')
    pdf.bullet_point('Feature importance reveals pricing drivers')
    pdf.ln(2)

    pdf.subsection_title('Portfolio Analytics')
    pdf.bullet_point('Portfolio-level liquidity scoring for risk management')
    pdf.bullet_point('Rebalancing optimization based on trade costs')
    pdf.bullet_point('Stress testing under different market conditions')
    pdf.ln(2)

    pdf.subsection_title('Compliance & Transparency')
    pdf.bullet_point('SHAP explanations provide audit trail for trading decisions')
    pdf.bullet_point('Model persistence ensures reproducibility')
    pdf.ln(4)

    # Technical Stack
    pdf.section_title('Technical Stack')
    pdf.add_table(
        headers=['Category', 'Technologies'],
        data=[
            ['Languages', 'Python 3.11+'],
            ['ML Frameworks', 'XGBoost 2.0+, LightGBM 4.0+, scikit-learn'],
            ['Explainability', 'SHAP (SHapley Additive exPlanations)'],
            ['Data Processing', 'pandas, numpy'],
            ['Visualization', 'Streamlit, matplotlib'],
            ['Data Sources', 'FRED API, SEC EDGAR, Yahoo Finance'],
        ],
        col_widths=[50, 140]
    )

    # Next Steps
    pdf.section_title('Production Readiness & Next Steps')
    pdf.bullet_point('API Integration: RESTful API wrapper for model serving')
    pdf.bullet_point('Real Data: Integration with live market data feeds')
    pdf.bullet_point('Model Monitoring: Drift detection and automated retraining')
    pdf.bullet_point('Scalability: Batch prediction pipeline for portfolio-level analysis')

    # Save the PDF
    output_dir = Path(__file__).parent
    output_path = output_dir / 'project_summary.pdf'
    pdf.output(str(output_path))

    return output_path


if __name__ == '__main__':
    output_path = generate_report()
    print(f"Report generated: {output_path}")
