"""
XGBoost Liquidity Score Model.

Provides a multi-class classifier for predicting loan liquidity tiers (1-5).
Uses XGBoost with SHAP explanations and time-series cross-validation.

Liquidity tiers represent trading ease:
    1 = Most liquid (frequent trading, tight spreads)
    5 = Least liquid (infrequent trading, wide spreads)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiquidityScoreModel:
    """
    XGBoost classifier for predicting loan liquidity tiers (1-5).

    This model uses gradient boosting to classify loans into liquidity tiers
    based on loan characteristics, market conditions, and trading metrics.
    Includes SHAP-based explanations for model interpretability.

    Attributes:
        model: Trained XGBClassifier instance.
        feature_names: List of feature names used during training.
        n_classes: Number of liquidity tier classes (5).
        is_fitted: Whether the model has been trained.

    Example:
        >>> model = LiquidityScoreModel(n_estimators=100)
        >>> cv_results = model.fit(X_train, y_train, cv_folds=5)
        >>> predictions = model.predict(X_test)
        >>> preds, shap_values = model.predict_with_explanation(X_test)
    """

    # Liquidity tier labels
    N_CLASSES = 5
    TIER_LABELS = {1: 'Most Liquid', 2: 'Liquid', 3: 'Moderate',
                   4: 'Illiquid', 5: 'Most Illiquid'}

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the LiquidityScoreModel.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate (eta).
            random_state: Random seed for reproducibility.
            **kwargs: Additional XGBoost parameters.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.extra_params = kwargs

        # Initialize the XGBoost classifier
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=self.N_CLASSES,
            random_state=random_state,
            eval_metric='mlogloss',
            **kwargs
        )

        # State tracking
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self._shap_explainer: Optional[shap.TreeExplainer] = None

        logger.info(
            f"LiquidityScoreModel initialized: n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate}"
        )

    def _validate_target(self, y: pd.Series) -> np.ndarray:
        """
        Validate and convert target variable to 0-indexed classes.

        XGBoost expects class labels starting from 0, but our liquidity
        tiers are 1-5. This method handles the conversion.

        Args:
            y: Target series with liquidity tiers (1-5).

        Returns:
            NumPy array with 0-indexed class labels (0-4).

        Raises:
            ValueError: If target contains invalid values.
        """
        unique_values = set(y.unique())
        expected_values = {1, 2, 3, 4, 5}

        if not unique_values.issubset(expected_values):
            invalid = unique_values - expected_values
            raise ValueError(
                f"Target contains invalid liquidity tier values: {invalid}. "
                f"Expected values in range 1-5."
            )

        # Convert 1-5 to 0-4 for XGBoost
        return (y - 1).values

    def _convert_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert 0-indexed predictions back to liquidity tiers (1-5).

        Args:
            predictions: Model predictions (0-4).

        Returns:
            Predictions as liquidity tiers (1-5).
        """
        return predictions + 1

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Train the model with time-series cross-validation.

        Uses TimeSeriesSplit to respect temporal ordering of data,
        which is important for financial time series to avoid
        look-ahead bias.

        Args:
            X: Feature DataFrame.
            y: Target Series with liquidity tiers (1-5).
            cv_folds: Number of cross-validation folds.
            eval_set: Optional (X_val, y_val) tuple for early stopping.

        Returns:
            Dictionary containing:
                - cv_scores: List of accuracy scores per fold
                - mean_cv_score: Mean cross-validation accuracy
                - std_cv_score: Standard deviation of CV scores
                - train_score: Final training accuracy
                - classification_report: Per-class metrics

        Raises:
            ValueError: If input data is invalid.
        """
        logger.info(f"Starting model training with {len(X)} samples, {len(X.columns)} features")

        # Store feature names for later use
        self.feature_names = list(X.columns)

        # Prepare target variable (convert 1-5 to 0-4)
        y_converted = self._validate_target(y)

        # Prepare feature matrix
        X_values = X.values

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        fold_reports = []

        logger.info(f"Running {cv_folds}-fold time-series cross-validation...")

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_values)):
            X_train_fold = X_values[train_idx]
            y_train_fold = y_converted[train_idx]
            X_val_fold = X_values[val_idx]
            y_val_fold = y_converted[val_idx]

            # Create a fresh model for this fold
            fold_model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective='multi:softprob',
                num_class=self.N_CLASSES,
                random_state=self.random_state,
                eval_metric='mlogloss',
                **self.extra_params
            )

            # Train fold model
            fold_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

            # Evaluate on validation fold
            y_pred_fold = fold_model.predict(X_val_fold)
            fold_accuracy = accuracy_score(y_val_fold, y_pred_fold)
            cv_scores.append(fold_accuracy)

            logger.info(f"  Fold {fold_idx + 1}/{cv_folds}: accuracy = {fold_accuracy:.4f}")

        # Calculate CV statistics
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        logger.info(f"CV Complete: mean accuracy = {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")

        # Train final model on full dataset
        logger.info("Training final model on full dataset...")

        if eval_set is not None:
            X_eval, y_eval = eval_set
            y_eval_converted = self._validate_target(y_eval)
            self.model.fit(
                X_values, y_converted,
                eval_set=[(X_eval.values, y_eval_converted)],
                verbose=False
            )
        else:
            self.model.fit(X_values, y_converted, verbose=False)

        self.is_fitted = True

        # Calculate training accuracy
        y_train_pred = self.model.predict(X_values)
        train_accuracy = accuracy_score(y_converted, y_train_pred)

        # Generate classification report (convert back to 1-5 for display)
        y_pred_display = self._convert_predictions(y_train_pred)
        y_display = y.values
        report = classification_report(
            y_display, y_pred_display,
            labels=[1, 2, 3, 4, 5],
            target_names=[f"Tier {i}" for i in range(1, 6)],
            output_dict=True
        )

        # Initialize SHAP explainer
        self._shap_explainer = shap.TreeExplainer(self.model)

        logger.info(f"Training complete: train accuracy = {train_accuracy:.4f}")

        return {
            'cv_scores': cv_scores,
            'mean_cv_score': mean_cv_score,
            'std_cv_score': std_cv_score,
            'train_score': train_accuracy,
            'classification_report': report,
            'n_features': len(self.feature_names),
            'n_samples': len(X)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict liquidity tiers (1-5) for input samples.

        Args:
            X: Feature DataFrame with same columns as training data.

        Returns:
            NumPy array of predicted liquidity tiers (1-5).

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If input features don't match training features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Validate feature alignment
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Ensure consistent column ordering
            X = X[self.feature_names]

        # Get predictions (0-4) and convert to tiers (1-5)
        predictions = self.model.predict(X.values)
        return self._convert_predictions(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability distribution over liquidity tiers.

        Returns probabilities for each class (tier 1-5), useful for
        understanding prediction confidence and near-boundary cases.

        Args:
            X: Feature DataFrame.

        Returns:
            NumPy array of shape (n_samples, 5) with class probabilities.
            Column i contains probability of tier i+1.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Validate and order features
        if self.feature_names is not None:
            X = X[self.feature_names]

        return self.model.predict_proba(X.values)

    def predict_with_explanation(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with SHAP value explanations.

        Returns both predictions and SHAP values explaining feature
        contributions to each prediction.

        Args:
            X: Feature DataFrame.

        Returns:
            Tuple of (predictions, shap_values):
                - predictions: Array of predicted tiers (1-5)
                - shap_values: Array of shape (n_samples, n_features, n_classes)
                              containing SHAP values for each feature and class

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Validate and order features
        if self.feature_names is not None:
            X = X[self.feature_names]

        # Get predictions
        predictions = self.predict(X)

        # Calculate SHAP values
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)

        shap_values = self._shap_explainer.shap_values(X.values)

        # shap_values is a list of arrays for multi-class
        # Convert to (n_samples, n_features, n_classes) array
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)

        return predictions, shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get sorted feature importance scores.

        Uses XGBoost's built-in feature importance (gain-based).

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - importance: Importance score
                - rank: Importance rank (1 = most important)

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")

        # Get importance scores
        importance_scores = self.model.feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)

        return importance_df

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_distributions: Optional[Dict] = None,
        n_iter: int = 50,
        cv_folds: int = 5,
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using randomized search with CV.

        Searches over a grid of hyperparameters to find the best
        configuration for the dataset.

        Args:
            X: Feature DataFrame.
            y: Target Series with liquidity tiers (1-5).
            param_distributions: Dictionary of parameter distributions.
                                If None, uses default search space.
            n_iter: Number of random parameter combinations to try.
            cv_folds: Number of cross-validation folds.
            n_jobs: Number of parallel jobs (-1 = all cores).
            verbose: Verbosity level (0-2).

        Returns:
            Dictionary containing:
                - best_params: Best hyperparameter combination
                - best_score: Best cross-validation score
                - cv_results: Full cross-validation results
        """
        logger.info(f"Starting hyperparameter tuning with {n_iter} iterations...")

        # Default parameter search space
        if param_distributions is None:
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3, 0.5],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0, 0.1, 1, 10]
            }

        # Prepare target (convert 1-5 to 0-4)
        y_converted = self._validate_target(y)

        # Store feature names
        self.feature_names = list(X.columns)

        # Create base estimator
        base_model = XGBClassifier(
            objective='multi:softprob',
            num_class=self.N_CLASSES,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Randomized search
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=tscv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state,
            return_train_score=True
        )

        # Fit the search
        search.fit(X.values, y_converted)

        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")

        # Update model with best parameters
        self.model = search.best_estimator_
        self.n_estimators = search.best_params_.get('n_estimators', self.n_estimators)
        self.max_depth = search.best_params_.get('max_depth', self.max_depth)
        self.learning_rate = search.best_params_.get('learning_rate', self.learning_rate)
        self.is_fitted = True

        # Initialize SHAP explainer with tuned model
        self._shap_explainer = shap.TreeExplainer(self.model)

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Saves the complete model state including:
        - Trained XGBoost model
        - Feature names
        - Hyperparameters
        - Fitted state

        Args:
            path: File path for saved model (typically .joblib extension).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = {
            'model': self.model,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'extra_params': self.extra_params,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LiquidityScoreModel':
        """
        Load a saved model from disk.

        Args:
            path: File path to the saved model.

        Returns:
            LiquidityScoreModel instance with restored state.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model state
        model_state = joblib.load(path)

        # Create new instance
        instance = cls(
            n_estimators=model_state['n_estimators'],
            max_depth=model_state['max_depth'],
            learning_rate=model_state['learning_rate'],
            random_state=model_state['random_state'],
            **model_state.get('extra_params', {})
        )

        # Restore state
        instance.model = model_state['model']
        instance.feature_names = model_state['feature_names']
        instance.is_fitted = model_state['is_fitted']

        # Reinitialize SHAP explainer if fitted
        if instance.is_fitted:
            instance._shap_explainer = shap.TreeExplainer(instance.model)

        logger.info(f"Model loaded from {path}")
        return instance

    def get_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate and return the confusion matrix as a DataFrame.

        Args:
            X: Feature DataFrame.
            y: True liquidity tier labels (1-5).

        Returns:
            Confusion matrix as a DataFrame with tier labels.
        """
        predictions = self.predict(X)
        cm = confusion_matrix(y.values, predictions, labels=[1, 2, 3, 4, 5])

        return pd.DataFrame(
            cm,
            index=[f"True Tier {i}" for i in range(1, 6)],
            columns=[f"Pred Tier {i}" for i in range(1, 6)]
        )


def print_model_summary(model: LiquidityScoreModel, cv_results: Dict) -> None:
    """
    Print a comprehensive summary of model training results.

    Args:
        model: Trained LiquidityScoreModel instance.
        cv_results: Cross-validation results from model.fit().
    """
    print("=" * 70)
    print("LIQUIDITY SCORE MODEL - TRAINING SUMMARY")
    print("=" * 70)

    print("\nModel Configuration:")
    print(f"  n_estimators:    {model.n_estimators}")
    print(f"  max_depth:       {model.max_depth}")
    print(f"  learning_rate:   {model.learning_rate}")
    print(f"  n_features:      {cv_results.get('n_features', 'N/A')}")
    print(f"  n_samples:       {cv_results.get('n_samples', 'N/A')}")

    print("\nCross-Validation Results:")
    print(f"  Mean CV Accuracy:  {cv_results['mean_cv_score']:.4f}")
    print(f"  Std CV Accuracy:   {cv_results['std_cv_score']:.4f}")
    print(f"  Training Accuracy: {cv_results['train_score']:.4f}")

    print("\nPer-Fold CV Scores:")
    for i, score in enumerate(cv_results['cv_scores'], 1):
        print(f"    Fold {i}: {score:.4f}")

    print("\nPer-Class Metrics:")
    report = cv_results['classification_report']
    print(f"  {'Tier':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 50)
    for tier in range(1, 6):
        tier_key = f"Tier {tier}"
        if tier_key in report:
            metrics = report[tier_key]
            print(f"  {tier_key:<10} {metrics['precision']:>10.3f} "
                  f"{metrics['recall']:>10.3f} {metrics['f1-score']:>10.3f} "
                  f"{int(metrics['support']):>10d}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    """
    Demonstration of LiquidityScoreModel training and prediction.

    This demo:
    1. Loads synthetic loan data
    2. Engineers features using available feature engines
    3. Trains the XGBoost model
    4. Evaluates performance
    5. Demonstrates SHAP explanations
    """
    from pathlib import Path
    import sys

    # Add parent directories to path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.features.loan_features import LoanFeatureEngine
    from src.features.liquidity_features import LiquidityFeatureEngine

    print("=" * 70)
    print("LIQUIDITY SCORE MODEL - DEMO")
    print("=" * 70)

    # Load synthetic data
    data_path = project_root / "data" / "synthetic_loans.csv"
    print(f"\n1. Loading data from: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"   Loaded {len(df_raw):,} loans")

    # Display target distribution
    print("\n   Liquidity tier distribution:")
    tier_counts = df_raw['liquidity_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = 100 * count / len(df_raw)
        label = LiquidityScoreModel.TIER_LABELS.get(tier, '')
        print(f"     Tier {tier} ({label:15s}): {count:,} ({pct:.1f}%)")

    # Apply feature engineering
    print("\n2. Engineering features...")
    loan_engine = LoanFeatureEngine()
    liquidity_engine = LiquidityFeatureEngine()

    # Apply loan features
    df_features = loan_engine.transform(df_raw)

    # Apply liquidity features
    df_features = liquidity_engine.transform(df_features)

    # Select numeric features for modeling
    # Exclude non-numeric columns and identifiers
    exclude_cols = {'loan_id', 'liquidity_tier', 'maturity_bucket'}
    feature_cols = [col for col in df_features.columns
                    if col not in exclude_cols
                    and df_features[col].dtype in ['int64', 'float64', 'bool']]

    X = df_features[feature_cols].copy()
    y = df_raw['liquidity_tier']

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples:  {len(X):,}")

    # Initialize and train model
    print("\n3. Training LiquidityScoreModel...")
    model = LiquidityScoreModel(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    cv_results = model.fit(X, y, cv_folds=5)

    # Print training summary
    print_model_summary(model, cv_results)

    # Feature importance
    print("\n4. Top 10 Most Important Features:")
    importance = model.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        print(f"   {row['rank']:2d}. {row['feature']:30s} {row['importance']:.4f}")

    # Make predictions on first 5 samples
    print("\n5. Sample Predictions:")
    sample_X = X.head(5)
    sample_y = y.head(5)
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)

    print(f"   {'Loan':<12} {'True':>6} {'Pred':>6} {'Confidence':>12}")
    print("   " + "-" * 40)
    for i, (true_tier, pred_tier) in enumerate(zip(sample_y, predictions)):
        max_prob = probabilities[i].max()
        match = "ok" if true_tier == pred_tier else "MISS"
        loan_id = df_raw.iloc[i]['loan_id']
        print(f"   {loan_id:<12} {true_tier:>6} {pred_tier:>6} {max_prob:>10.1%}  {match}")

    # SHAP explanations for first sample
    print("\n6. SHAP Explanation (first sample):")
    preds, shap_values = model.predict_with_explanation(sample_X.iloc[[0]])
    predicted_class = int(preds[0] - 1)  # Convert to 0-indexed

    print(f"   Predicted tier: {preds[0]} ({model.TIER_LABELS[preds[0]]})")
    print("\n   Top contributing features:")

    # Get SHAP values for predicted class
    sample_shap = shap_values[0, :, predicted_class]
    feature_shap = list(zip(feature_cols, sample_shap))
    feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    for feature, value in feature_shap[:5]:
        direction = "+" if value > 0 else ""
        print(f"     {feature:30s}: {direction}{value:.4f}")

    # Confusion matrix
    print("\n7. Confusion Matrix (full dataset):")
    cm = model.get_confusion_matrix(X, y)
    print(cm.to_string())

    # Save and reload demo
    print("\n8. Model Persistence Demo:")
    save_path = project_root / "models" / "liquidity_model_demo.joblib"
    model.save(str(save_path))
    print(f"   Saved to: {save_path}")

    loaded_model = LiquidityScoreModel.load(str(save_path))
    reload_preds = loaded_model.predict(sample_X)
    print(f"   Reloaded and verified: predictions match = {np.array_equal(predictions, reload_preds)}")

    # Overall assessment
    print("\n" + "=" * 70)
    accuracy = cv_results['mean_cv_score']
    target_met = accuracy >= 0.70
    status = "PASSED" if target_met else "BELOW TARGET"
    print(f"TARGET: >70% accuracy | ACHIEVED: {accuracy:.1%} | STATUS: {status}")
    print("=" * 70)
