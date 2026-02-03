"""
CLO Loan-Level Liquidity Predictor - Streamlit Dashboard.

Interactive dashboard for exploring loan liquidity predictions using:
- LiquidityScoreModel (XGBoost classifier for tiers 1-5)
- TradeCostPredictor (LightGBM regressor for bid-ask spread)
- SHAPExplainer for model interpretability

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.liquidity_model import LiquidityScoreModel
from src.models.spread_model import TradeCostPredictor
from src.explainability.shap_utils import SHAPExplainer
from src.features.loan_features import LoanFeatureEngine
from src.features.liquidity_features import LiquidityFeatureEngine

# Page configuration
st.set_page_config(
    page_title="CLO Liquidity Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tier-1 { background-color: #28a745; color: white; }
    .tier-2 { background-color: #5cb85c; color: white; }
    .tier-3 { background-color: #ffc107; color: black; }
    .tier-4 { background-color: #fd7e14; color: white; }
    .tier-5 { background-color: #dc3545; color: white; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CREDIT_RATINGS = ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC']
INDUSTRY_SECTORS = ['Technology', 'Healthcare', 'Energy', 'Consumer',
                    'Industrials', 'Financials', 'Telecom', 'Utilities']
TIER_LABELS = {1: 'Most Liquid', 2: 'Liquid', 3: 'Moderate',
               4: 'Illiquid', 5: 'Most Illiquid'}
TIER_COLORS = {1: '#28a745', 2: '#5cb85c', 3: '#ffc107',
               4: '#fd7e14', 5: '#dc3545'}

DATA_PATH = project_root / "data" / "synthetic_loans.csv"
LIQUIDITY_MODEL_PATH = project_root / "models" / "liquidity_model.joblib"
SPREAD_MODEL_PATH = project_root / "models" / "spread_model.joblib"


@st.cache_data
def load_synthetic_data():
    """Load synthetic loan data from CSV."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_or_train_models():
    """
    Load pre-trained models if available, otherwise train on synthetic data.
    Returns trained models and feature-engineered data.
    """
    # Load data
    df_raw = load_synthetic_data()
    if df_raw is None:
        st.error("Synthetic data file not found. Please ensure data/synthetic_loans.csv exists.")
        return None, None, None, None

    # Apply feature engineering
    loan_engine = LoanFeatureEngine()
    liquidity_engine = LiquidityFeatureEngine()

    df_features = loan_engine.transform(df_raw)
    df_features = liquidity_engine.transform(df_features)

    # Prepare features for models
    exclude_cols = {'loan_id', 'liquidity_tier', 'maturity_bucket'}
    feature_cols = [col for col in df_features.columns
                    if col not in exclude_cols
                    and df_features[col].dtype in ['int64', 'float64', 'bool']]

    X = df_features[feature_cols].copy()

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    y_tier = df_raw['liquidity_tier']
    y_spread = df_raw['bid_ask_spread']

    # Try to load or train liquidity model
    liquidity_model = None
    try:
        if LIQUIDITY_MODEL_PATH.exists():
            liquidity_model = LiquidityScoreModel.load(str(LIQUIDITY_MODEL_PATH))
        else:
            liquidity_model = LiquidityScoreModel(n_estimators=100, max_depth=6)
            liquidity_model.fit(X, y_tier, cv_folds=3)
            # Save for future use
            LIQUIDITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            liquidity_model.save(str(LIQUIDITY_MODEL_PATH))
    except Exception as e:
        st.warning(f"Training liquidity model on the fly: {e}")
        liquidity_model = LiquidityScoreModel(n_estimators=100, max_depth=6)
        liquidity_model.fit(X, y_tier, cv_folds=3)

    # Try to load or train spread model
    spread_model = None
    try:
        if SPREAD_MODEL_PATH.exists():
            spread_model = TradeCostPredictor.load(str(SPREAD_MODEL_PATH))
        else:
            spread_model = TradeCostPredictor(n_estimators=100, max_depth=6)
            spread_model.fit(X, y_spread)
            spread_model.save(str(SPREAD_MODEL_PATH))
    except Exception as e:
        st.warning(f"Training spread model on the fly: {e}")
        spread_model = TradeCostPredictor(n_estimators=100, max_depth=6)
        spread_model.fit(X, y_spread)

    return liquidity_model, spread_model, X, df_raw


def prepare_input_features(params: dict, reference_X: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare input features from sidebar parameters to match training features.

    Args:
        params: Dictionary of loan parameters from sidebar
        reference_X: Reference DataFrame with correct feature columns

    Returns:
        DataFrame with properly formatted features for prediction
    """
    # Create raw loan data matching the expected input format
    raw_data = pd.DataFrame([{
        'loan_id': 'USER_INPUT',
        'facility_size': params['facility_size'],
        'credit_rating': params['credit_rating'],
        'current_spread': params['current_spread'],
        'time_to_maturity': params['time_to_maturity'],
        'industry_sector': params['industry_sector'],
        'trading_volume_30d': params['trading_volume_30d'],
        'bid_ask_spread': params['bid_ask_spread'],
        'covenant_lite': params['covenant_lite']
    }])

    # Apply feature engineering
    loan_engine = LoanFeatureEngine()
    liquidity_engine = LiquidityFeatureEngine()

    df_features = loan_engine.transform(raw_data)
    df_features = liquidity_engine.transform(df_features)

    # Prepare features matching training columns
    exclude_cols = {'loan_id', 'liquidity_tier', 'maturity_bucket'}
    feature_cols = [col for col in reference_X.columns if col in df_features.columns]

    X_input = df_features[feature_cols].copy()

    # Convert boolean columns to int
    bool_cols = X_input.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X_input[col] = X_input[col].astype(int)

    # Ensure all expected columns are present
    for col in reference_X.columns:
        if col not in X_input.columns:
            X_input[col] = 0

    # Reorder columns to match training data
    X_input = X_input[reference_X.columns]

    return X_input


def render_sidebar():
    """Render sidebar with loan input parameters."""
    st.sidebar.header("Loan Parameters")
    st.sidebar.markdown("Configure the loan characteristics below:")

    st.sidebar.subheader("Size & Structure")
    facility_size = st.sidebar.slider(
        "Facility Size ($M)",
        min_value=50, max_value=2000, value=500, step=50,
        help="Total facility size in millions of dollars"
    )

    covenant_lite = st.sidebar.checkbox(
        "Covenant-Lite", value=True,
        help="Whether the loan has limited covenants"
    )

    st.sidebar.subheader("Credit & Spread")
    credit_rating = st.sidebar.selectbox(
        "Credit Rating",
        options=CREDIT_RATINGS, index=3,
        help="Credit rating from BB+ (best) to CCC (highest risk)"
    )

    current_spread = st.sidebar.slider(
        "Current Spread (bps)",
        min_value=150, max_value=900, value=350, step=25,
        help="Spread over SOFR in basis points"
    )

    st.sidebar.subheader("Maturity & Sector")
    time_to_maturity = st.sidebar.slider(
        "Time to Maturity (years)",
        min_value=0.5, max_value=7.0, value=4.0, step=0.5,
        help="Years until loan maturity"
    )

    industry_sector = st.sidebar.selectbox(
        "Industry Sector",
        options=INDUSTRY_SECTORS, index=0,
        help="Industry classification of the borrower"
    )

    st.sidebar.subheader("Trading Metrics")
    trading_volume_30d = st.sidebar.slider(
        "Trading Volume 30d ($M)",
        min_value=1, max_value=100, value=20, step=1,
        help="30-day trading volume in millions"
    )

    bid_ask_spread = st.sidebar.slider(
        "Current Bid-Ask (bps)",
        min_value=20, max_value=250, value=100, step=5,
        help="Current bid-ask spread in basis points"
    )

    return {
        'facility_size': facility_size,
        'credit_rating': credit_rating,
        'current_spread': current_spread,
        'time_to_maturity': time_to_maturity,
        'industry_sector': industry_sector,
        'trading_volume_30d': trading_volume_30d,
        'bid_ask_spread': bid_ask_spread,
        'covenant_lite': covenant_lite
    }


def render_predictions_tab(liquidity_model, spread_model, X_input, params):
    """Render the Predictions tab content."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Liquidity Tier Prediction")

        # Get predictions
        tier_pred = liquidity_model.predict(X_input)[0]
        tier_probs = liquidity_model.predict_proba(X_input)[0]

        # Display predicted tier with color coding
        tier_color = TIER_COLORS[tier_pred]
        tier_label = TIER_LABELS[tier_pred]

        st.markdown(f"""
        <div style="background-color: {tier_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">Tier {tier_pred}</h1>
            <p style="color: white; margin: 0; font-size: 1.2rem;">{tier_label}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability distribution bar chart
        prob_df = pd.DataFrame({
            'Tier': [f"Tier {i}" for i in range(1, 6)],
            'Probability': tier_probs,
            'Label': [TIER_LABELS[i] for i in range(1, 6)]
        })

        fig = px.bar(
            prob_df, x='Tier', y='Probability',
            color='Tier',
            color_discrete_map={f"Tier {i}": TIER_COLORS[i] for i in range(1, 6)},
            title="Tier Probability Distribution",
            hover_data=['Label']
        )
        fig.update_layout(
            showlegend=False,
            yaxis_tickformat='.0%',
            yaxis_title="Probability",
            xaxis_title="Liquidity Tier"
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Confidence indicator
        max_prob = tier_probs.max()
        confidence_text = "High" if max_prob > 0.6 else "Medium" if max_prob > 0.4 else "Low"
        st.metric("Prediction Confidence", f"{max_prob:.1%}", confidence_text)

    with col2:
        st.subheader("Bid-Ask Spread Prediction")

        # Get spread prediction with confidence interval
        spread_pred, spread_lower, spread_upper = spread_model.predict_with_confidence(
            X_input, n_bootstrap=100, confidence_level=0.95
        )
        spread_pred = spread_pred[0]
        spread_lower = spread_lower[0]
        spread_upper = spread_upper[0]

        # Display predicted spread
        st.markdown(f"""
        <div style="background-color: #1f4e79; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">{spread_pred:.0f} bps</h1>
            <p style="color: white; margin: 0; font-size: 1rem;">Predicted Bid-Ask Spread</p>
            <p style="color: #ccc; margin: 5px 0 0 0; font-size: 0.9rem;">
                95% CI: [{spread_lower:.0f} - {spread_upper:.0f}] bps
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence interval visualization
        fig = go.Figure()

        # Add confidence interval
        fig.add_trace(go.Bar(
            x=['Spread'],
            y=[spread_upper - spread_lower],
            base=[spread_lower],
            marker_color='lightblue',
            name='95% CI',
            width=0.4
        ))

        # Add point estimate
        fig.add_trace(go.Scatter(
            x=['Spread'],
            y=[spread_pred],
            mode='markers',
            marker=dict(size=20, color='#1f4e79', symbol='diamond'),
            name='Predicted'
        ))

        # Add current spread for comparison
        fig.add_trace(go.Scatter(
            x=['Spread'],
            y=[params['bid_ask_spread']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Current'
        ))

        fig.update_layout(
            title="Spread Prediction vs Current",
            yaxis_title="Bid-Ask Spread (bps)",
            showlegend=True,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade cost estimation
        st.subheader("Trade Cost Estimation")
        trade_sizes = [1, 5, 10, 25, 50]

        cost_data = []
        for size in trade_sizes:
            notional = size * 1_000_000
            cost = spread_model.calculate_trade_cost(spread_pred, notional)
            cost_data.append({
                'Trade Size ($M)': size,
                'Est. Cost ($)': f"${cost:,.0f}",
                'Cost (bps)': f"{spread_pred/2:.0f}"
            })

        cost_df = pd.DataFrame(cost_data)
        st.dataframe(cost_df, use_container_width=True, hide_index=True)


def render_explainability_tab(liquidity_model, spread_model, X_input, reference_X):
    """Render the Explainability tab content."""
    st.subheader("Model Explainability")

    model_choice = st.radio(
        "Select model to explain:",
        ["Liquidity Tier Model", "Spread Prediction Model"],
        horizontal=True
    )

    if model_choice == "Liquidity Tier Model":
        model = liquidity_model.model
        tier_pred = liquidity_model.predict(X_input)[0]

        st.markdown(f"**Explaining prediction for Tier {tier_pred} ({TIER_LABELS[tier_pred]})**")
    else:
        model = spread_model.model
        spread_pred = spread_model.predict(X_input)[0]

        st.markdown(f"**Explaining prediction for {spread_pred:.0f} bps spread**")

    # Create SHAP explainer
    try:
        explainer = SHAPExplainer(model, model_type='tree')

        # Get feature importance
        importance_df = explainer.get_feature_importance(X_input)

        # Feature importance bar chart
        st.subheader("Feature Importance")

        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)

        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Top {top_n} Features by SHAP Importance",
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Feature",
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Single prediction explanation
        st.subheader("Prediction Breakdown")

        if model_choice == "Liquidity Tier Model":
            class_idx = tier_pred - 1  # Convert to 0-indexed
        else:
            class_idx = None

        explanation = explainer.explain_single_prediction(
            X_input, idx=0, top_n=10, class_index=class_idx
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Positive Contributors**")
            st.markdown("*(pushing prediction higher)*")

            for feat, val in explanation['top_positive'][:5]:
                feat_val = explanation['feature_values'].get(feat, 'N/A')
                if isinstance(feat_val, (int, float)):
                    feat_val = f"{feat_val:.2f}"
                st.markdown(f"- **{feat}** = {feat_val} (+{val:.4f})")

        with col2:
            st.markdown("**Top Negative Contributors**")
            st.markdown("*(pushing prediction lower)*")

            for feat, val in explanation['top_negative'][:5]:
                feat_val = explanation['feature_values'].get(feat, 'N/A')
                if isinstance(feat_val, (int, float)):
                    feat_val = f"{feat_val:.2f}"
                st.markdown(f"- **{feat}** = {feat_val} ({val:.4f})")

        # Interpretation text
        st.subheader("Interpretation")

        if model_choice == "Liquidity Tier Model":
            if tier_pred <= 2:
                st.success("""
                This loan is predicted to be **highly liquid**. Key drivers include:
                - Higher trading volume relative to peers
                - Tighter bid-ask spreads
                - Stronger credit profile

                Expect frequent trading opportunities and competitive pricing.
                """)
            elif tier_pred == 3:
                st.info("""
                This loan shows **moderate liquidity**. It may:
                - Trade less frequently than the most liquid names
                - Have wider bid-ask spreads
                - Require more time to execute larger trades

                Consider breaking up large orders into smaller pieces.
                """)
            else:
                st.warning("""
                This loan is predicted to be **relatively illiquid**. Characteristics include:
                - Lower trading activity
                - Wider bid-ask spreads
                - Fewer dealer quotes

                Plan for extended execution time and potential price impact on larger trades.
                """)
        else:
            if explanation['prediction'] < 80:
                st.success("""
                The predicted spread is **relatively tight**, suggesting good liquidity.
                Transaction costs should be manageable for most trade sizes.
                """)
            elif explanation['prediction'] < 150:
                st.info("""
                The predicted spread is **moderate**. Factor in transaction costs when
                evaluating expected returns, especially for frequent trading strategies.
                """)
            else:
                st.warning("""
                The predicted spread is **wide**, indicating challenging liquidity conditions.
                Consider the significant transaction costs when planning trades.
                Patient execution strategies may help reduce costs.
                """)

    except Exception as e:
        st.error(f"Error generating explanations: {str(e)}")
        st.info("Try adjusting the loan parameters and rerunning.")


def render_data_explorer_tab(df_raw, liquidity_model, reference_X):
    """Render the Data Explorer tab content."""
    st.subheader("Synthetic Loan Dataset Explorer")

    if df_raw is None:
        st.error("No data available to explore.")
        return

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Loans", f"{len(df_raw):,}")
    with col2:
        st.metric("Avg Facility Size", f"${df_raw['facility_size'].mean():.0f}M")
    with col3:
        st.metric("Avg Spread", f"{df_raw['current_spread'].mean():.0f} bps")
    with col4:
        st.metric("Avg Bid-Ask", f"{df_raw['bid_ask_spread'].mean():.0f} bps")

    st.markdown("---")

    # Filters
    st.subheader("Filter Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_tiers = st.multiselect(
            "Liquidity Tiers",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5],
            format_func=lambda x: f"Tier {x} - {TIER_LABELS[x]}"
        )

    with col2:
        selected_ratings = st.multiselect(
            "Credit Ratings",
            options=CREDIT_RATINGS,
            default=CREDIT_RATINGS
        )

    with col3:
        selected_sectors = st.multiselect(
            "Industry Sectors",
            options=df_raw['industry_sector'].unique().tolist(),
            default=df_raw['industry_sector'].unique().tolist()
        )

    # Apply filters
    mask = (
        (df_raw['liquidity_tier'].isin(selected_tiers)) &
        (df_raw['credit_rating'].isin(selected_ratings)) &
        (df_raw['industry_sector'].isin(selected_sectors))
    )
    df_filtered = df_raw[mask]

    st.info(f"Showing {len(df_filtered):,} of {len(df_raw):,} loans")

    # Visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Tier distribution
        tier_counts = df_filtered['liquidity_tier'].value_counts().sort_index()
        fig = px.bar(
            x=[f"Tier {i}" for i in tier_counts.index],
            y=tier_counts.values,
            color=[f"Tier {i}" for i in tier_counts.index],
            color_discrete_map={f"Tier {i}": TIER_COLORS[i] for i in range(1, 6)},
            title="Liquidity Tier Distribution"
        )
        fig.update_layout(showlegend=False, xaxis_title="Tier", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with viz_col2:
        # Rating distribution
        fig = px.pie(
            df_filtered, names='credit_rating',
            title="Credit Rating Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plots
    st.subheader("Relationship Explorer")

    scatter_col1, scatter_col2 = st.columns(2)

    with scatter_col1:
        x_var = st.selectbox(
            "X-axis variable",
            options=['facility_size', 'trading_volume_30d', 'current_spread', 'time_to_maturity'],
            index=0
        )

    with scatter_col2:
        y_var = st.selectbox(
            "Y-axis variable",
            options=['bid_ask_spread', 'trading_volume_30d', 'current_spread', 'facility_size'],
            index=0
        )

    color_by = st.radio(
        "Color by:",
        ["Liquidity Tier", "Credit Rating", "Industry Sector"],
        horizontal=True
    )

    color_col = {
        "Liquidity Tier": "liquidity_tier",
        "Credit Rating": "credit_rating",
        "Industry Sector": "industry_sector"
    }[color_by]

    fig = px.scatter(
        df_filtered,
        x=x_var,
        y=y_var,
        color=color_col,
        color_discrete_map={i: TIER_COLORS[i] for i in range(1, 6)} if color_by == "Liquidity Tier" else None,
        hover_data=['loan_id', 'credit_rating', 'industry_sector'],
        title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
        opacity=0.7
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Data Table")

    display_cols = ['loan_id', 'facility_size', 'credit_rating', 'current_spread',
                    'time_to_maturity', 'industry_sector', 'trading_volume_30d',
                    'bid_ask_spread', 'covenant_lite', 'liquidity_tier']

    st.dataframe(
        df_filtered[display_cols].head(100),
        use_container_width=True,
        hide_index=True
    )

    if len(df_filtered) > 100:
        st.caption(f"Showing first 100 of {len(df_filtered)} filtered loans")


def main():
    """Main application entry point."""
    # Header
    st.markdown('<p class="main-header">CLO Loan-Level Liquidity Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict loan liquidity tiers and trading costs with ML models</p>', unsafe_allow_html=True)

    # Load models and data
    with st.spinner("Loading models and data..."):
        liquidity_model, spread_model, reference_X, df_raw = load_or_train_models()

    if liquidity_model is None or spread_model is None:
        st.error("Failed to load or train models. Please check the data files.")
        return

    # Sidebar inputs
    params = render_sidebar()

    # Prepare input features
    X_input = prepare_input_features(params, reference_X)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🔍 Explainability", "📊 Data Explorer"])

    with tab1:
        render_predictions_tab(liquidity_model, spread_model, X_input, params)

    with tab2:
        render_explainability_tab(liquidity_model, spread_model, X_input, reference_X)

    with tab3:
        render_data_explorer_tab(df_raw, liquidity_model, reference_X)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>CLO Loan-Level Liquidity Predictor | Built with Streamlit</p>
        <p>Models: XGBoost (Liquidity Tier) | LightGBM (Bid-Ask Spread) | SHAP (Explainability)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
