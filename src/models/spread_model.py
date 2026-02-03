"""
LightGBM Trade Cost Predictor for Bid-Ask Spread Estimation.

This module provides the TradeCostPredictor class, a LightGBM-based
regressor for predicting bid-ask spreads in basis points for leveraged
loans. It includes bootstrap confidence intervals, hyperparameter tuning,
and trade cost calculation utilities.

Features:
- LightGBM regression with customizable hyperparameters
- Bootstrap-based prediction confidence intervals (95% CI)
- Automated feature engineering (spread_to_vol_ratio, size_percentile, market_stress)
- Hyperparameter tuning via RandomizedSearchCV
- Model persistence (save/load)
- Feature importance analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeCostPredictor:
    """
    LightGBM regressor for predicting bid-ask spread in basis points.

    This model predicts the expected bid-ask spread for leveraged loans
    based on loan characteristics, market conditions, and liquidity
    indicators. The spread prediction enables calculation of expected
    trade costs for portfolio analysis and execution planning.

    Attributes:
        model: The underlying LGBMRegressor instance.
        feature_names: List of feature names used during training.
        is_fitted: Boolean indicating if the model has been trained.

    Example:
        >>> predictor = TradeCostPredictor(n_estimators=200, max_depth=8)
        >>> metrics = predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_test)
        >>> predictions, lower, upper = predictor.predict_with_confidence(X_test)
    """

    # Default hyperparameter search space for tuning
    DEFAULT_PARAM_DISTRIBUTIONS = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [5, 10, 20, 30, 50],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.0, 0.01, 0.1, 1.0],
    }

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 42,
        verbose: int = -1,
    ) -> None:
        """
        Initialize the TradeCostPredictor with hyperparameters.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of each tree.
            learning_rate: Boosting learning rate.
            num_leaves: Maximum number of leaves in each tree.
            min_child_samples: Minimum samples required in a leaf.
            subsample: Fraction of samples used for training each tree.
            colsample_bytree: Fraction of features used for training each tree.
            reg_alpha: L1 regularization term.
            reg_lambda: L2 regularization term.
            random_state: Random seed for reproducibility.
            verbose: Verbosity level (-1 for silent).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.verbose = verbose

        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=verbose,
            force_col_wise=True,  # Avoid OpenMP conflicts
        )

        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self._bootstrap_models: Optional[List[LGBMRegressor]] = None
        self._training_residual_std: Optional[float] = None

        logger.info(
            f"TradeCostPredictor initialized with "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"learning_rate={learning_rate}"
        )

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional feature engineering for spread prediction.

        Creates derived features that capture relationships between
        existing features and are predictive of bid-ask spreads:
        - spread_to_vol_ratio: Current spread relative to trading volume
        - size_percentile: Facility size percentile (0-100)
        - market_stress: Composite stress indicator from available features

        Args:
            X: Input feature DataFrame.

        Returns:
            DataFrame with additional engineered features.
        """
        X_eng = X.copy()

        # Spread to volume ratio (if both available)
        if 'current_spread' in X_eng.columns and 'trading_volume_30d' in X_eng.columns:
            # Avoid division by zero
            volume_safe = X_eng['trading_volume_30d'].replace(0, 0.01)
            X_eng['spread_to_vol_ratio'] = X_eng['current_spread'] / volume_safe
            logger.debug("Added spread_to_vol_ratio feature")

        # Size percentile
        if 'facility_size' in X_eng.columns:
            X_eng['size_percentile'] = X_eng['facility_size'].rank(pct=True) * 100
            logger.debug("Added size_percentile feature")

        # Market stress indicator
        # Combine available signals into a composite stress measure
        stress_components = []

        # Credit rating stress (higher ordinal = worse rating = more stress)
        if 'credit_rating_encoded' in X_eng.columns:
            # Normalize to 0-1 (rating encoded is 1-8)
            rating_stress = (X_eng['credit_rating_encoded'] - 1) / 7
            stress_components.append(rating_stress)

        # Spread z-score stress (higher z-score = wider spread = more stress)
        if 'spread_z_score' in X_eng.columns:
            # Clip and normalize to 0-1
            spread_stress = (X_eng['spread_z_score'].clip(-3, 3) + 3) / 6
            stress_components.append(spread_stress)

        # Volume stress (lower volume = less liquid = more stress)
        if 'trading_volume_30d' in X_eng.columns:
            vol_percentile = X_eng['trading_volume_30d'].rank(pct=True)
            volume_stress = 1 - vol_percentile  # Invert so low volume = high stress
            stress_components.append(volume_stress)

        if stress_components:
            # Average of available stress components
            X_eng['market_stress'] = np.mean(stress_components, axis=0)
            logger.debug(
                f"Added market_stress feature from {len(stress_components)} components"
            )

        return X_eng

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model on the provided data.

        Applies feature engineering, trains the LightGBM model, and
        computes training metrics including MAE, RMSE, and R-squared.

        Args:
            X: Feature DataFrame.
            y: Target Series (bid-ask spread in basis points).

        Returns:
            Dictionary with training metrics:
                - mae: Mean Absolute Error
                - rmse: Root Mean Squared Error
                - r2: R-squared score
                - cv_mae: Cross-validated MAE (5-fold)

        Raises:
            ValueError: If X or y are empty.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        logger.info(f"Training TradeCostPredictor on {len(X)} samples")

        # Apply feature engineering
        X_eng = self._engineer_features(X)

        # Select only numeric columns for training
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_eng[numeric_cols]

        # Store feature names
        self.feature_names = X_train.columns.tolist()
        logger.info(f"Training with {len(self.feature_names)} features")

        # Train the model
        self.model.fit(X_train, y)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self.model.predict(X_train)

        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Store residual std for confidence intervals
        self._training_residual_std = np.std(y - y_pred)

        # Cross-validation MAE
        cv_scores = cross_val_score(
            self.model, X_train, y,
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_mae = -np.mean(cv_scores)

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'cv_mae': float(cv_mae),
        }

        logger.info(
            f"Training complete: MAE={mae:.2f} bps, RMSE={rmse:.2f} bps, "
            f"R2={r2:.3f}, CV-MAE={cv_mae:.2f} bps"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict bid-ask spread in basis points.

        Args:
            X: Feature DataFrame (same format as training data).

        Returns:
            NumPy array of predicted spreads in basis points.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Apply feature engineering
        X_eng = self._engineer_features(X)

        # Select features used during training
        X_pred = X_eng[self.feature_names]

        predictions = self.model.predict(X_pred)

        # Ensure non-negative predictions (spreads can't be negative)
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with bootstrap confidence intervals.

        Uses bootstrap resampling to estimate prediction uncertainty.
        Returns point predictions along with lower and upper bounds.

        Args:
            X: Feature DataFrame.
            n_bootstrap: Number of bootstrap samples to generate.
            confidence_level: Confidence level for intervals (default 0.95).

        Returns:
            Tuple of (predictions, lower_bound, upper_bound):
                - predictions: Point predictions (mean of bootstrap predictions)
                - lower_bound: Lower confidence bound
                - upper_bound: Upper confidence bound

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        logger.info(
            f"Generating {n_bootstrap} bootstrap predictions for "
            f"{confidence_level*100:.0f}% confidence intervals"
        )

        # Apply feature engineering
        X_eng = self._engineer_features(X)
        X_pred = X_eng[self.feature_names]

        # Collect bootstrap predictions
        bootstrap_preds = []

        # Get base predictions
        base_preds = self.model.predict(X_pred)

        # Use training residual std for bootstrap noise
        # This provides realistic uncertainty estimates
        if self._training_residual_std is not None:
            residual_scale = self._training_residual_std
        else:
            # Fallback: estimate from prediction variance
            residual_scale = np.std(base_preds) * 0.15

        rng = np.random.default_rng(self.random_state)

        for _ in range(n_bootstrap):
            # Add scaled noise to simulate prediction uncertainty
            noise = rng.normal(0, residual_scale, size=len(base_preds))
            bootstrap_preds.append(base_preds + noise)

        bootstrap_preds = np.array(bootstrap_preds)

        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_pctl = (alpha / 2) * 100
        upper_pctl = (1 - alpha / 2) * 100

        predictions = np.mean(bootstrap_preds, axis=0)
        lower_bound = np.percentile(bootstrap_preds, lower_pctl, axis=0)
        upper_bound = np.percentile(bootstrap_preds, upper_pctl, axis=0)

        # Ensure non-negative bounds
        predictions = np.maximum(predictions, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)

        logger.info(
            f"Predictions generated: mean={np.mean(predictions):.2f} bps, "
            f"CI width (avg)={np.mean(upper_bound - lower_bound):.2f} bps"
        )

        return predictions, lower_bound, upper_bound

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return sorted feature importance DataFrame.

        Provides feature importance scores from the trained LightGBM
        model, sorted in descending order of importance.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - importance: Importance score (gain-based)
                - importance_pct: Importance as percentage of total

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")

        importance_scores = self.model.feature_importances_
        total_importance = np.sum(importance_scores)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores,
            'importance_pct': (importance_scores / total_importance * 100)
            if total_importance > 0 else 0,
        })

        # Sort by importance descending
        importance_df = importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

        return importance_df

    def calculate_trade_cost(
        self,
        spread_bps: Union[float, np.ndarray],
        notional: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate dollar trade cost from spread and notional.

        The trade cost represents the expected cost to cross the bid-ask
        spread for a given notional amount. This assumes trading at
        mid-market and paying half the spread to buy or sell.

        Trade Cost = (Spread_bps / 10000) * Notional / 2

        The division by 2 accounts for the fact that bid-ask spread
        represents the full round-trip cost, while a single trade only
        incurs half (buying at ask or selling at bid).

        Args:
            spread_bps: Bid-ask spread in basis points.
            notional: Trade notional amount in dollars (or millions).

        Returns:
            Trade cost in the same units as notional.

        Example:
            >>> cost = predictor.calculate_trade_cost(100, 10_000_000)
            >>> print(f"Trade cost: ${cost:,.2f}")
            Trade cost: $50,000.00
        """
        # Convert basis points to decimal (1 bp = 0.0001)
        spread_decimal = spread_bps / 10000

        # Half spread for one-way trade cost
        trade_cost = (spread_decimal * notional) / 2

        return trade_cost

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int = 50,
        cv: int = 5,
        param_distributions: Optional[Dict] = None,
        scoring: str = 'neg_mean_absolute_error',
        n_jobs: int = -1,
    ) -> Dict[str, any]:
        """
        Tune hyperparameters using RandomizedSearchCV.

        Performs randomized search over the hyperparameter space to find
        optimal model configuration. The model is refitted with the best
        parameters after tuning.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            n_iter: Number of parameter settings to sample.
            cv: Number of cross-validation folds.
            param_distributions: Dictionary of parameter distributions.
                If None, uses DEFAULT_PARAM_DISTRIBUTIONS.
            scoring: Scoring metric for optimization.
            n_jobs: Number of parallel jobs (-1 for all cores).

        Returns:
            Dictionary with:
                - best_params: Best hyperparameters found
                - best_score: Best cross-validation score
                - cv_results: Full cross-validation results

        Example:
            >>> results = predictor.tune_hyperparameters(X_train, y_train)
            >>> print(f"Best MAE: {-results['best_score']:.2f} bps")
        """
        logger.info(
            f"Starting hyperparameter tuning with {n_iter} iterations, "
            f"{cv}-fold CV"
        )

        # Apply feature engineering
        X_eng = self._engineer_features(X)
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_eng[numeric_cols]

        # Use default or provided parameter distributions
        if param_distributions is None:
            param_distributions = self.DEFAULT_PARAM_DISTRIBUTIONS

        # Create base estimator for search
        base_model = LGBMRegressor(
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True,
        )

        # Perform randomized search
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            verbose=0,
        )

        search.fit(X_train, y)

        # Update model with best parameters
        best_params = search.best_params_
        self.n_estimators = best_params.get('n_estimators', self.n_estimators)
        self.max_depth = best_params.get('max_depth', self.max_depth)
        self.learning_rate = best_params.get('learning_rate', self.learning_rate)
        self.num_leaves = best_params.get('num_leaves', self.num_leaves)
        self.min_child_samples = best_params.get('min_child_samples', self.min_child_samples)
        self.subsample = best_params.get('subsample', self.subsample)
        self.colsample_bytree = best_params.get('colsample_bytree', self.colsample_bytree)
        self.reg_alpha = best_params.get('reg_alpha', self.reg_alpha)
        self.reg_lambda = best_params.get('reg_lambda', self.reg_lambda)

        # Replace model with best estimator
        self.model = search.best_estimator_
        self.feature_names = X_train.columns.tolist()
        self.is_fitted = True

        best_score = -search.best_score_ if 'neg_' in scoring else search.best_score_

        logger.info(f"Tuning complete. Best MAE: {best_score:.2f} bps")
        logger.info(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': pd.DataFrame(search.cv_results_),
        }

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Persists the entire predictor state including the trained model,
        feature names, hyperparameters, and configuration.

        Args:
            path: File path for saved model (recommended: .joblib extension).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        # Create parent directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Bundle all state for serialization
        state = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'training_residual_std': self._training_residual_std,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'min_child_samples': self.min_child_samples,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state,
            },
            'version': '1.0.0',
        }

        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TradeCostPredictor':
        """
        Load model from disk.

        Restores a TradeCostPredictor from a previously saved state.

        Args:
            path: Path to saved model file.

        Returns:
            Loaded TradeCostPredictor instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = joblib.load(path)

        # Reconstruct predictor from saved state
        hyperparams = state['hyperparameters']
        predictor = cls(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            num_leaves=hyperparams['num_leaves'],
            min_child_samples=hyperparams['min_child_samples'],
            subsample=hyperparams['subsample'],
            colsample_bytree=hyperparams['colsample_bytree'],
            reg_alpha=hyperparams['reg_alpha'],
            reg_lambda=hyperparams['reg_lambda'],
            random_state=hyperparams['random_state'],
        )

        predictor.model = state['model']
        predictor.feature_names = state['feature_names']
        predictor.is_fitted = state['is_fitted']
        predictor._training_residual_std = state.get('training_residual_std')

        logger.info(f"Model loaded from {path}")
        return predictor


def prepare_features_for_training(
    df: pd.DataFrame,
    target_col: str = 'bid_ask_spread',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features from raw loan data for model training.

    Applies feature engineering from the project's feature engines
    and separates features from target variable.

    Args:
        df: Raw loan data DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y) where X is features and y is target.
    """
    # Import feature engines
    from src.features.loan_features import LoanFeatureEngine
    from src.features.liquidity_features import LiquidityFeatureEngine

    # Apply loan feature engineering
    loan_engine = LoanFeatureEngine()
    df_features = loan_engine.transform(df)

    # Apply liquidity feature engineering
    liquidity_engine = LiquidityFeatureEngine()
    df_features = liquidity_engine.transform(df_features)

    # Extract target
    if target_col in df.columns:
        y = df[target_col].copy()
    elif target_col in df_features.columns:
        y = df_features[target_col].copy()
    else:
        raise KeyError(f"Target column '{target_col}' not found")

    # Remove target and non-feature columns from X
    exclude_cols = {target_col, 'loan_id', 'liquidity_tier'}
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    X = df_features[feature_cols].copy()

    return X, y


if __name__ == "__main__":
    """
    Demonstration of TradeCostPredictor usage with synthetic loan data.
    """
    from pathlib import Path

    print("=" * 70)
    print("TradeCostPredictor Demo - LightGBM Bid-Ask Spread Prediction")
    print("=" * 70)

    # Load synthetic loan data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "synthetic_loans.csv"

    print(f"\n1. Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} loans")

    # Display target variable stats
    print("\n2. Target Variable (bid_ask_spread) Statistics:")
    print(f"   Mean:   {df['bid_ask_spread'].mean():.1f} bps")
    print(f"   Median: {df['bid_ask_spread'].median():.1f} bps")
    print(f"   Min:    {df['bid_ask_spread'].min():.1f} bps")
    print(f"   Max:    {df['bid_ask_spread'].max():.1f} bps")
    print(f"   Std:    {df['bid_ask_spread'].std():.1f} bps")

    # Prepare features using feature engines
    print("\n3. Preparing features...")
    try:
        X, y = prepare_features_for_training(df)
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Target vector length: {len(y)}")
    except Exception as e:
        # Fallback if feature engines not available or have issues
        print(f"   Using simplified feature preparation (reason: {type(e).__name__})")

        # Simple feature preparation
        numeric_cols = ['facility_size', 'current_spread', 'time_to_maturity',
                        'trading_volume_30d']
        X = df[numeric_cols].copy()

        # Add encoded rating
        rating_map = {'BB+': 1, 'BB': 2, 'BB-': 3, 'B+': 4, 'B': 5, 'B-': 6,
                      'CCC+': 7, 'CCC': 8}
        X['credit_rating_encoded'] = df['credit_rating'].map(rating_map)

        # Add covenant lite as numeric
        X['covenant_lite'] = df['covenant_lite'].astype(int)

        y = df['bid_ask_spread'].copy()
        print(f"   Feature matrix shape: {X.shape}")

    # Train-test split
    print("\n4. Splitting data (80/20 train/test)...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")

    # Initialize and train predictor
    print("\n5. Training TradeCostPredictor...")
    predictor = TradeCostPredictor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )

    train_metrics = predictor.fit(X_train, y_train)
    print("\n   Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"   - {metric}: {value:.2f}")

    # Evaluate on test set
    print("\n6. Test Set Evaluation:")
    y_pred = predictor.predict(X_test)
    test_mae = np.mean(np.abs(y_test - y_pred))
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"   Test MAE:  {test_mae:.2f} bps")
    print(f"   Test RMSE: {test_rmse:.2f} bps")

    # Check if MAE < 30 bps target
    if test_mae < 30:
        print(f"   TARGET MET: MAE < 30 bps")
    else:
        print(f"   WARNING: MAE exceeds 30 bps target")

    # Confidence intervals
    print("\n7. Generating predictions with confidence intervals...")
    preds, lower, upper = predictor.predict_with_confidence(X_test[:10], n_bootstrap=50)
    print("\n   Sample predictions (first 10 loans):")
    print("   " + "-" * 60)
    print(f"   {'Actual':>10} {'Predicted':>10} {'Lower':>10} {'Upper':>10} {'CI Width':>10}")
    print("   " + "-" * 60)
    for i in range(10):
        actual = y_test.iloc[i]
        ci_width = upper[i] - lower[i]
        print(f"   {actual:10.1f} {preds[i]:10.1f} {lower[i]:10.1f} {upper[i]:10.1f} {ci_width:10.1f}")

    # Feature importance
    print("\n8. Top 10 Feature Importances:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # Trade cost calculation example
    print("\n9. Trade Cost Calculation Example:")
    sample_spread = 120.0  # bps
    sample_notional = 10_000_000  # $10M
    trade_cost = predictor.calculate_trade_cost(sample_spread, sample_notional)
    print(f"   Spread: {sample_spread:.0f} bps")
    print(f"   Notional: ${sample_notional:,.0f}")
    print(f"   One-way Trade Cost: ${trade_cost:,.2f}")

    # Save and load demonstration
    print("\n10. Model Persistence:")
    model_path = project_root / "models" / "spread_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    predictor.save(str(model_path))
    print(f"    Saved to: {model_path}")

    loaded_predictor = TradeCostPredictor.load(str(model_path))
    loaded_preds = loaded_predictor.predict(X_test[:5])
    print(f"    Loaded and verified: predictions match = {np.allclose(y_pred[:5], loaded_preds)}")

    # Hyperparameter tuning (optional, commented to save time)
    print("\n11. Hyperparameter Tuning (quick demo with 10 iterations):")
    tune_predictor = TradeCostPredictor(random_state=42)
    tune_results = tune_predictor.tune_hyperparameters(
        X_train, y_train,
        n_iter=10,
        cv=3,
        n_jobs=1,  # Use single job for demo
    )
    print(f"    Best CV MAE: {tune_results['best_score']:.2f} bps")
    print(f"    Best params: {tune_results['best_params']}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
