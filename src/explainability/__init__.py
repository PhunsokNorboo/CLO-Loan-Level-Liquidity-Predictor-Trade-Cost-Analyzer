"""
Explainability module for loan liquidity predictor.

Provides model interpretability using SHAP:
- Global feature importance
- Local explanations for individual predictions
- Visualization utilities
- HTML report generation

Classes:
    SHAPExplainer: Unified SHAP explainability for tree-based models.

Functions:
    plot_liquidity_tier_explanation: Custom visualization for liquidity tier predictions.
    generate_explanation_report: Generate comprehensive HTML explanation reports.

Example:
    >>> from explainability import SHAPExplainer
    >>> explainer = SHAPExplainer(trained_model, model_type='tree')
    >>> importance = explainer.get_feature_importance(X)
    >>> explainer.summary_plot(X, plot_type='bar')
"""

from src.explainability.shap_utils import (
    SHAPExplainer,
    generate_explanation_report,
    plot_liquidity_tier_explanation,
)

__all__ = [
    'SHAPExplainer',
    'generate_explanation_report',
    'plot_liquidity_tier_explanation',
]
