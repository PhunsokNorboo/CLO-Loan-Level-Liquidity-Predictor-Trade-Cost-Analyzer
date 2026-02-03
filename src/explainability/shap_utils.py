"""
SHAP Explainability Utilities for Loan Liquidity Predictor.

Provides unified SHAP-based model interpretability for tree-based models
(XGBoost, LightGBM). Supports both classification and regression tasks with:
- Local explanations for individual predictions
- Global feature importance analysis
- Publication-quality visualizations
- HTML report generation

Usage:
    >>> from explainability.shap_utils import SHAPExplainer
    >>> explainer = SHAPExplainer(model, model_type='tree')
    >>> explanation = explainer.explain(X)
    >>> explainer.waterfall_plot(X, idx=0)
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """
    Unified SHAP explainability for tree-based models.

    Provides a consistent interface for computing SHAP values and generating
    visualizations for both XGBoost classifiers and LightGBM regressors.

    Attributes:
        model: The trained model to explain.
        model_type: Type of SHAP explainer to use ('tree' or 'linear').
        explainer: The underlying SHAP explainer instance.
        is_classifier: Whether the model is a classifier (vs regressor).
        n_classes: Number of classes for classification models.

    Example:
        >>> import xgboost as xgb
        >>> model = xgb.XGBClassifier().fit(X_train, y_train)
        >>> explainer = SHAPExplainer(model, model_type='tree')
        >>> importance = explainer.get_feature_importance(X_test)
        >>> explainer.summary_plot(X_test, plot_type='bar')
    """

    def __init__(self, model: Any, model_type: str = 'tree') -> None:
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained model (XGBoost or LightGBM). Must have a predict
                   method. Classifiers should also have predict_proba.
            model_type: Type of explainer to use:
                        - 'tree': TreeExplainer for tree-based models (default)
                        - 'linear': LinearExplainer for linear models

        Raises:
            ValueError: If model_type is not supported.
            RuntimeError: If explainer initialization fails.
        """
        self.model = model
        self.model_type = model_type.lower()

        # Detect if model is a classifier
        self.is_classifier = hasattr(model, 'predict_proba')
        self.n_classes: Optional[int] = None

        # Attempt to determine number of classes for classifiers
        if self.is_classifier:
            if hasattr(model, 'n_classes_'):
                self.n_classes = model.n_classes_
            elif hasattr(model, 'classes_'):
                self.n_classes = len(model.classes_)

        # Initialize the appropriate SHAP explainer
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, shap.sample(model))
        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Use 'tree' or 'linear'."
            )

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Calculate SHAP values for input data.

        Computes SHAP values explaining how each feature contributes to
        predictions. For multi-class classification, returns values for
        each class.

        Args:
            X: Input features as a pandas DataFrame. Column names are
               preserved in the explanation object.

        Returns:
            shap.Explanation object containing:
                - values: SHAP values (n_samples x n_features for regression,
                          n_samples x n_features x n_classes for classification)
                - base_values: Expected model output (base prediction)
                - data: Original feature values
                - feature_names: Column names from X

        Note:
            For multi-class classifiers, SHAP values have an additional
            dimension for class probabilities. Use the class_index parameter
            in visualization methods to select a specific class.
        """
        # Compute SHAP values
        shap_values = self.explainer(X)

        # Ensure feature names are preserved
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is None:
            shap_values.feature_names = list(X.columns)

        return shap_values

    def waterfall_plot(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        max_display: int = 10,
        show: bool = True,
        class_index: Optional[int] = None
    ) -> plt.Figure:
        """
        Generate waterfall plot for a single prediction.

        Shows how each feature pushes the prediction from the base value
        (expected model output) to the final prediction.

        Args:
            X: Input features DataFrame.
            idx: Index of the sample to explain (0-based).
            max_display: Maximum number of features to display.
            show: If True, display the plot immediately.
            class_index: For multi-class models, which class to explain.
                         Defaults to the predicted class.

        Returns:
            matplotlib Figure object.

        Raises:
            IndexError: If idx is out of bounds for X.
        """
        if idx >= len(X):
            raise IndexError(f"Index {idx} out of bounds for data with {len(X)} samples")

        shap_values = self.explain(X)

        # Handle multi-class classification
        if self.is_classifier and self.n_classes is not None and self.n_classes > 2:
            if class_index is None:
                # Default to predicted class
                predictions = self.model.predict(X)
                class_index = predictions[idx]
            sample_shap = shap_values[idx, :, class_index]
        else:
            sample_shap = shap_values[idx]

        # Create figure
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sample_shap, max_display=max_display, show=False)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def force_plot(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        class_index: Optional[int] = None
    ) -> Any:
        """
        Generate force plot for a single prediction.

        Creates an interactive visualization showing feature contributions
        as forces pushing the prediction from the base value.

        Args:
            X: Input features DataFrame.
            idx: Index of the sample to explain (0-based).
            class_index: For multi-class models, which class to explain.

        Returns:
            SHAP force plot object (can be displayed in Jupyter notebooks
            or saved as HTML).

        Note:
            Force plots require JavaScript to render interactively.
            In non-notebook environments, use matplotlib=True parameter
            for static rendering.
        """
        shap_values = self.explain(X)

        # Handle multi-class classification
        if self.is_classifier and self.n_classes is not None and self.n_classes > 2:
            if class_index is None:
                predictions = self.model.predict(X)
                class_index = predictions[idx]
            sample_shap = shap_values[idx, :, class_index]
        else:
            sample_shap = shap_values[idx]

        # Create force plot
        force = shap.plots.force(sample_shap)

        return force

    def summary_plot(
        self,
        X: pd.DataFrame,
        plot_type: str = 'bar',
        max_display: int = 15,
        show: bool = True,
        class_index: Optional[int] = None
    ) -> plt.Figure:
        """
        Generate summary plot showing feature importance.

        Creates either a bar chart of mean absolute SHAP values or a
        beeswarm plot showing the distribution of SHAP values per feature.

        Args:
            X: Input features DataFrame.
            plot_type: Type of summary plot:
                       - 'bar': Mean absolute SHAP values (feature importance)
                       - 'dot' or 'beeswarm': Individual SHAP value distributions
            max_display: Maximum number of features to display.
            show: If True, display the plot immediately.
            class_index: For multi-class models, which class to explain.
                         If None for bar plots, shows overall importance.

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: If plot_type is not 'bar', 'dot', or 'beeswarm'.
        """
        valid_plot_types = {'bar', 'dot', 'beeswarm'}
        if plot_type not in valid_plot_types:
            raise ValueError(f"plot_type must be one of {valid_plot_types}")

        shap_values = self.explain(X)

        # Handle multi-class classification
        # SHAP plots work better with 2D arrays, so we either select a class
        # or compute mean absolute values across classes
        if self.is_classifier and self.n_classes is not None and self.n_classes > 2:
            if class_index is not None:
                # Use specific class
                shap_values = shap_values[:, :, class_index]
            elif plot_type == 'bar':
                # For bar plots, compute mean absolute across classes
                # Create a new Explanation with averaged values
                mean_abs_values = np.abs(shap_values.values).mean(axis=2)
                shap_values = shap.Explanation(
                    values=mean_abs_values,
                    base_values=shap_values.base_values.mean(axis=1),
                    data=shap_values.data,
                    feature_names=shap_values.feature_names
                )
            else:
                # For beeswarm, use class 0 by default
                shap_values = shap_values[:, :, 0]

        # Create figure
        fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))

        if plot_type == 'bar':
            shap.plots.bar(shap_values, max_display=max_display, show=False)
        else:
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def dependence_plot(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        show: bool = True,
        class_index: Optional[int] = None
    ) -> plt.Figure:
        """
        Generate dependence plot for a specific feature.

        Shows the relationship between a feature's value and its SHAP value,
        optionally colored by an interaction feature.

        Args:
            X: Input features DataFrame.
            feature: Name of the feature to plot.
            interaction_feature: Feature to use for coloring points. If None,
                                 SHAP automatically selects the strongest
                                 interaction.
            show: If True, display the plot immediately.
            class_index: For multi-class models, which class to explain.

        Returns:
            matplotlib Figure object.

        Raises:
            KeyError: If feature is not found in X.
        """
        if feature not in X.columns:
            raise KeyError(f"Feature '{feature}' not found in input data")

        shap_values = self.explain(X)

        # Handle multi-class classification
        if self.is_classifier and self.n_classes is not None and self.n_classes > 2:
            if class_index is None:
                class_index = 0
            shap_values = shap_values[:, :, class_index]

        # Create figure
        fig = plt.figure(figsize=(10, 6))

        shap.plots.scatter(
            shap_values[:, feature],
            color=shap_values[:, interaction_feature] if interaction_feature else None,
            show=False
        )

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP values for feature importance.

        Computes the average absolute SHAP value for each feature across
        all samples, providing a global measure of feature importance.

        Args:
            X: Input features DataFrame.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - importance: Mean absolute SHAP value
                - importance_pct: Percentage of total importance
            Sorted by importance in descending order.
        """
        shap_values = self.explain(X)

        # Extract SHAP values array
        if self.is_classifier and self.n_classes is not None and self.n_classes > 2:
            # For multi-class, average across classes
            values = np.abs(shap_values.values).mean(axis=2)
        else:
            values = np.abs(shap_values.values)

        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.mean(values, axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_abs_shap
        })

        # Calculate percentage
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = (
                importance_df['importance'] / total_importance * 100
            )
        else:
            importance_df['importance_pct'] = 0.0

        # Sort by importance
        importance_df = importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

        return importance_df

    def explain_single_prediction(
        self,
        X: pd.DataFrame,
        idx: int = 0,
        top_n: int = 5,
        class_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a single prediction.

        Provides a comprehensive breakdown of how features contribute
        to a specific prediction, including top drivers in each direction.

        Args:
            X: Input features DataFrame.
            idx: Index of the sample to explain (0-based).
            top_n: Number of top positive/negative contributors to return.
            class_index: For multi-class models, which class to explain.
                         Defaults to the predicted class.

        Returns:
            Dictionary containing:
                - prediction: Model output (probability for classifiers)
                - base_value: Expected model output (baseline)
                - feature_contributions: Dict mapping feature -> SHAP value
                - top_positive: List of (feature, value) tuples for features
                                pushing prediction up
                - top_negative: List of (feature, value) tuples for features
                                pushing prediction down
                - feature_values: Dict mapping feature -> input value

        Raises:
            IndexError: If idx is out of bounds for X.
        """
        if idx >= len(X):
            raise IndexError(f"Index {idx} out of bounds for data with {len(X)} samples")

        # Get prediction
        sample = X.iloc[[idx]]
        if self.is_classifier:
            prediction = self.model.predict_proba(sample)
            if self.n_classes is not None and self.n_classes > 2:
                if class_index is None:
                    class_index = self.model.predict(sample)[0]
                prediction = prediction[0][class_index]
            else:
                prediction = prediction[0][1]  # Probability of positive class
        else:
            prediction = self.model.predict(sample)[0]

        # Get SHAP values
        shap_values = self.explain(X)

        # Handle multi-class
        if self.is_classifier and self.n_classes is not None and self.n_classes > 2:
            if class_index is None:
                class_index = self.model.predict(sample)[0]
            sample_shap = shap_values[idx, :, class_index]
            base_value = shap_values.base_values[idx, class_index]
        else:
            sample_shap = shap_values[idx]
            base_value = shap_values.base_values[idx]
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0] if len(base_value) > 0 else base_value

        # Extract SHAP values
        shap_array = sample_shap.values
        feature_names = list(X.columns)

        # Create feature contributions dict
        feature_contributions = {
            name: float(shap_array[i])
            for i, name in enumerate(feature_names)
        }

        # Sort for top positive and negative
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_positive = [
            (name, value) for name, value in sorted_contributions
            if value > 0
        ][:top_n]

        top_negative = [
            (name, value) for name, value in sorted_contributions
            if value < 0
        ][-top_n:]
        top_negative.reverse()  # Most negative first

        # Get feature values
        feature_values = {
            name: X.iloc[idx][name]
            for name in feature_names
        }

        return {
            'prediction': float(prediction),
            'base_value': float(base_value),
            'feature_contributions': feature_contributions,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'feature_values': feature_values
        }


def plot_liquidity_tier_explanation(
    model: Any,
    explainer: SHAPExplainer,
    X: pd.DataFrame,
    idx: int = 0,
    tier_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create custom visualization for liquidity tier prediction.

    Shows tier probabilities alongside top feature drivers for a
    specific loan prediction.

    Args:
        model: Trained classifier model.
        explainer: SHAPExplainer instance for the model.
        X: Input features DataFrame.
        idx: Index of sample to explain.
        tier_names: Names for liquidity tiers (e.g., ['Illiquid', 'Low', 'Medium', 'High']).

    Returns:
        matplotlib Figure with two subplots:
        - Left: Tier probability bar chart
        - Right: Top feature drivers (positive and negative)
    """
    # Get prediction probabilities
    probs = model.predict_proba(X.iloc[[idx]])[0]
    n_tiers = len(probs)

    # Default tier names
    if tier_names is None:
        tier_names = [f'Tier {i+1}' for i in range(n_tiers)]

    # Get single prediction explanation
    predicted_tier = model.predict(X.iloc[[idx]])[0]
    explanation = explainer.explain_single_prediction(
        X, idx=idx, top_n=5, class_index=predicted_tier
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: Tier probabilities
    colors = ['#d73027', '#fc8d59', '#91cf60', '#1a9850'][:n_tiers]
    bars = ax1.barh(tier_names, probs, color=colors)

    # Highlight predicted tier
    bars[predicted_tier].set_edgecolor('black')
    bars[predicted_tier].set_linewidth(2)

    ax1.set_xlabel('Probability')
    ax1.set_title(f'Liquidity Tier Prediction\nPredicted: {tier_names[predicted_tier]}')
    ax1.set_xlim(0, 1)

    # Add probability labels
    for bar, prob in zip(bars, probs):
        ax1.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{prob:.1%}', va='center', fontsize=10)

    # Right subplot: Feature drivers
    top_positive = explanation['top_positive']
    top_negative = explanation['top_negative']

    features = [f for f, _ in top_positive] + [f for f, _ in top_negative]
    values = [v for _, v in top_positive] + [v for _, v in top_negative]
    colors = ['#1a9850' if v > 0 else '#d73027' for v in values]

    y_pos = np.arange(len(features))
    ax2.barh(y_pos, values, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('SHAP Value (impact on prediction)')
    ax2.set_title(f'Top Feature Drivers for {tier_names[predicted_tier]}')

    plt.tight_layout()

    return fig


def generate_explanation_report(
    model: Any,
    explainer: SHAPExplainer,
    X: pd.DataFrame,
    output_path: str,
    model_name: str = 'Model',
    sample_indices: Optional[List[int]] = None
) -> None:
    """
    Generate HTML report with comprehensive model explanations.

    Creates a standalone HTML file containing:
    - Model summary statistics
    - Global feature importance (bar chart + table)
    - Sample individual explanations
    - Feature dependence plots for top features

    Args:
        model: Trained model.
        explainer: SHAPExplainer instance for the model.
        X: Input features DataFrame used for explanations.
        output_path: Path to save the HTML report.
        model_name: Name to display in the report header.
        sample_indices: Indices of samples to include in individual
                        explanations. Defaults to first 3 samples.

    Note:
        Matplotlib figures are embedded as base64-encoded PNG images.
    """
    import base64
    from io import BytesIO
    from datetime import datetime

    def fig_to_base64(fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    # Default sample indices
    if sample_indices is None:
        sample_indices = list(range(min(3, len(X))))

    # Calculate feature importance
    importance_df = explainer.get_feature_importance(X)

    # Generate plots
    summary_fig = explainer.summary_plot(X, plot_type='bar', max_display=15, show=False)
    summary_img = fig_to_base64(summary_fig)

    beeswarm_fig = explainer.summary_plot(X, plot_type='dot', max_display=15, show=False)
    beeswarm_img = fig_to_base64(beeswarm_fig)

    # Generate individual explanation plots
    individual_plots = []
    for idx in sample_indices:
        try:
            fig = explainer.waterfall_plot(X, idx=idx, max_display=10, show=False)
            individual_plots.append({
                'idx': idx,
                'img': fig_to_base64(fig)
            })
        except Exception as e:
            individual_plots.append({
                'idx': idx,
                'error': str(e)
            })

    # Generate top feature dependence plots
    top_features = importance_df.head(3)['feature'].tolist()
    dependence_plots = []
    for feature in top_features:
        try:
            fig = explainer.dependence_plot(X, feature=feature, show=False)
            dependence_plots.append({
                'feature': feature,
                'img': fig_to_base64(fig)
            })
        except Exception as e:
            dependence_plots.append({
                'feature': feature,
                'error': str(e)
            })

    # Build HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} - SHAP Explanation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f0f0f0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .importance-bar {{
            background: linear-gradient(to right, #4CAF50 0%, #4CAF50 var(--width), #f0f0f0 var(--width));
            height: 20px;
            border-radius: 4px;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
    </style>
</head>
<body>
    <h1>{model_name} - SHAP Explanation Report</h1>
    <p class="metadata">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
       Samples analyzed: {len(X):,} |
       Features: {len(X.columns)}</p>

    <div class="section">
        <h2>Global Feature Importance</h2>
        <p>Mean absolute SHAP values across all samples. Higher values indicate
           features with greater impact on model predictions.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{summary_img}" alt="Feature Importance Bar Chart">
        </div>

        <h3>Feature Importance Table</h3>
        <table>
            <tr>
                <th>Rank</th>
                <th>Feature</th>
                <th>Importance</th>
                <th>% of Total</th>
            </tr>
"""

    for i, row in importance_df.head(15).iterrows():
        html_content += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{row['feature']}</td>
                <td>{row['importance']:.4f}</td>
                <td>{row['importance_pct']:.1f}%</td>
            </tr>
"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>Feature Value Distributions</h2>
        <p>Beeswarm plot showing how each feature's value affects predictions.
           Colors indicate feature values (red = high, blue = low).</p>
        <div class="plot-container">
"""
    html_content += f'<img src="data:image/png;base64,{beeswarm_img}" alt="Beeswarm Plot">'

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Individual Prediction Explanations</h2>
        <p>Waterfall plots showing how features contribute to specific predictions.</p>
        <div class="grid">
"""

    for plot_data in individual_plots:
        if 'error' in plot_data:
            html_content += f"""
            <div>
                <h3>Sample {plot_data['idx']}</h3>
                <p style="color: red;">Error: {plot_data['error']}</p>
            </div>
"""
        else:
            html_content += f"""
            <div class="plot-container">
                <h3>Sample {plot_data['idx']}</h3>
                <img src="data:image/png;base64,{plot_data['img']}" alt="Waterfall Plot">
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Feature Dependence Plots</h2>
        <p>Relationship between feature values and their SHAP values for top features.</p>
        <div class="grid">
"""

    for plot_data in dependence_plots:
        if 'error' in plot_data:
            html_content += f"""
            <div>
                <h3>{plot_data['feature']}</h3>
                <p style="color: red;">Error: {plot_data['error']}</p>
            </div>
"""
        else:
            html_content += f"""
            <div class="plot-container">
                <h3>{plot_data['feature']}</h3>
                <img src="data:image/png;base64,{plot_data['img']}" alt="Dependence Plot">
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Interpretation Guide</h2>
        <ul>
            <li><strong>SHAP values</strong>: Measure each feature's contribution to moving
                the prediction from the base value (average prediction) to the actual prediction.</li>
            <li><strong>Positive SHAP values</strong>: Push the prediction higher (toward
                positive class for classifiers, higher values for regressors).</li>
            <li><strong>Negative SHAP values</strong>: Push the prediction lower.</li>
            <li><strong>Feature importance</strong>: Mean absolute SHAP value shows overall
                impact regardless of direction.</li>
        </ul>
    </div>
</body>
</html>
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Explanation report saved to: {output_path}")


if __name__ == "__main__":
    """
    Demonstration of SHAP explainability utilities.

    Creates a simple trained model and demonstrates all visualization methods.
    """
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 70)
    print("SHAP EXPLAINABILITY UTILITIES - DEMONSTRATION")
    print("=" * 70)

    # Generate synthetic data for demonstration
    print("\n1. Generating synthetic loan data...")
    np.random.seed(42)
    n_samples = 500

    # Feature data
    X = pd.DataFrame({
        'facility_size_log': np.random.normal(6, 0.5, n_samples),
        'credit_rating_encoded': np.random.randint(1, 9, n_samples),
        'current_spread': np.random.normal(400, 100, n_samples),
        'time_to_maturity': np.random.uniform(1, 7, n_samples),
        'volume_percentile': np.random.uniform(0, 100, n_samples),
        'bid_ask_spread': np.random.exponential(50, n_samples) + 20,
        'dealer_coverage': np.random.uniform(0.1, 0.6, n_samples),
        'clo_ownership_pct': np.random.uniform(40, 80, n_samples),
    })

    # Create target: liquidity tier (0=Low, 1=Medium, 2=High)
    # Based on volume, spread, and dealer coverage
    liquidity_score = (
        X['volume_percentile'] / 100 * 0.4 +
        (100 - X['bid_ask_spread']) / 100 * 0.3 +
        X['dealer_coverage'] * 0.3
    )
    y = pd.cut(liquidity_score, bins=3, labels=[0, 1, 2]).astype(int)

    print(f"   Created {n_samples} samples with {len(X.columns)} features")
    print(f"   Target distribution: {dict(y.value_counts().sort_index())}")

    # Train XGBoost classifier
    print("\n2. Training XGBoost classifier...")
    try:
        import xgboost as xgb

        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]

        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)

        train_acc = (model.predict(X_train) == y_train).mean()
        test_acc = (model.predict(X_test) == y_test).mean()
        print(f"   Train accuracy: {train_acc:.2%}")
        print(f"   Test accuracy: {test_acc:.2%}")

    except ImportError:
        print("   XGBoost not available. Using LightGBM...")
        try:
            import lightgbm as lgb

            X_train, X_test = X[:400], X[400:]
            y_train, y_test = y[:400], y[400:]

            model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)

            train_acc = (model.predict(X_train) == y_train).mean()
            test_acc = (model.predict(X_test) == y_test).mean()
            print(f"   Train accuracy: {train_acc:.2%}")
            print(f"   Test accuracy: {test_acc:.2%}")

        except ImportError:
            print("   Neither XGBoost nor LightGBM available.")
            print("   Please install: pip install xgboost lightgbm shap")
            exit(1)

    # Initialize SHAP explainer
    print("\n3. Initializing SHAP explainer...")
    explainer = SHAPExplainer(model, model_type='tree')
    print(f"   Model type: {'classifier' if explainer.is_classifier else 'regressor'}")
    print(f"   Number of classes: {explainer.n_classes}")

    # Calculate feature importance
    print("\n4. Calculating feature importance...")
    importance_df = explainer.get_feature_importance(X_test)
    print("\n   Top features by mean |SHAP|:")
    for _, row in importance_df.head(5).iterrows():
        print(f"   - {row['feature']:25s}: {row['importance']:.4f} ({row['importance_pct']:.1f}%)")

    # Explain single prediction
    print("\n5. Explaining single prediction (sample 0)...")
    explanation = explainer.explain_single_prediction(X_test, idx=0)
    print(f"   Prediction: {explanation['prediction']:.4f}")
    print(f"   Base value: {explanation['base_value']:.4f}")
    print("\n   Top positive contributors:")
    for feat, val in explanation['top_positive'][:3]:
        print(f"   + {feat:25s}: {val:+.4f}")
    print("\n   Top negative contributors:")
    for feat, val in explanation['top_negative'][:3]:
        print(f"   - {feat:25s}: {val:+.4f}")

    # Generate plots (non-interactive mode)
    print("\n6. Generating visualization examples...")
    print("   (Plots saved but not displayed in headless mode)")

    # Summary bar plot
    fig = explainer.summary_plot(X_test, plot_type='bar', show=False)
    plt.close(fig)
    print("   - Summary bar plot: OK")

    # Waterfall plot
    fig = explainer.waterfall_plot(X_test, idx=0, show=False)
    plt.close(fig)
    print("   - Waterfall plot: OK")

    # Custom liquidity tier plot
    fig = plot_liquidity_tier_explanation(
        model, explainer, X_test, idx=0,
        tier_names=['Low Liquidity', 'Medium Liquidity', 'High Liquidity']
    )
    plt.close(fig)
    print("   - Liquidity tier explanation: OK")

    # Generate HTML report
    print("\n7. Generating HTML explanation report...")
    report_path = '/tmp/shap_explanation_report.html'
    generate_explanation_report(
        model, explainer, X_test,
        output_path=report_path,
        model_name='Loan Liquidity Classifier'
    )

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"\nHTML report saved to: {report_path}")
    print("\nUsage example:")
    print("  >>> from explainability.shap_utils import SHAPExplainer")
    print("  >>> explainer = SHAPExplainer(trained_model)")
    print("  >>> importance = explainer.get_feature_importance(X)")
    print("  >>> explainer.summary_plot(X)")
