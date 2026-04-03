"""
Modeling Agent - Builds and trains models for heating consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import logging
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from agents import BaseAgent

logger = logging.getLogger(__name__)


class ModelingAgent(BaseAgent):
    """Agent responsible for building and training models to predict heating consumption.
    
    The model predicts the heating component of energy consumption based on:
    - Temperature (main driver)
    - Time-based patterns (hour, day, season)
    - Baseline consumption levels
    
    Heating consumption = Total consumption - Baseline (non-heating) consumption
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("modeling_agent", config)
        self.models_path = Path(self.config.get("models_path", "models"))
        self.models_path.mkdir(exist_ok=True)
        # Preferred data resolution for analysis
        self.preferred_resolution = self.config.get("preferred_resolution", "quarterly")
        self.best_model = None
        self.model_performance = {}

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Build and train models on the processed data.

        Args:
            data: Dictionary of DataFrames with heating detection labels.

        Returns:
            Dictionary containing trained models and performance metrics.
        """
        results = {}

        # Find consumption data with heating labels - prefer specified resolution
        consumption_data = None
        resolution_priority = {
            'two_minutes': 'consumption_two_minutes.csv',
            'quarterly': 'consumption_quarterly.csv',
            'hourly': 'consumption_hourly.csv'
        }
        
        # Try to find preferred resolution first
        preferred_name = resolution_priority.get(self.preferred_resolution)
        if preferred_name and preferred_name in data:
            if 'heating_detected' in data[preferred_name].columns:
                consumption_data = data[preferred_name]
                logger.info(f"Using {preferred_name} (preferred resolution: {self.preferred_resolution})")
        
        # Fallback: find any consumption data with heating labels
        if consumption_data is None:
            for name, df in data.items():
                if 'consumption' in name.lower() and 'heating_detected' in df.columns:
                    consumption_data = df
                    logger.info(f"Using {name} for modeling")
                    break

        if consumption_data is None:
            logger.error("No consumption data with heating labels found")
            return results

        try:
            # Prepare data for modeling
            X, y = self._prepare_modeling_data(consumption_data)

            if X is None or y is None:
                logger.error("Failed to prepare modeling data")
                return results

            # Train models
            models = self._train_models(X, y)
            results['models'] = models

            # Evaluate models
            performance = self._evaluate_models(models, X, y)
            results['performance'] = performance

            # Select best model
            self.best_model = self._select_best_model(models, performance)
            results['best_model'] = self.best_model

            # Save models
            self._save_models(models)

            logger.info("Modeling completed successfully")

        except Exception as e:
            logger.error(f"Modeling failed: {e}")

        return results

    def _prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for modeling heating consumption prediction.
        
        Target: Heating consumption = Total consumption - Baseline (non-heating) consumption
        Features: Temperature, time patterns, and related indicators
        """
        # Find consumption and baseline columns
        consumption_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['consumption', 'energy', 'kwh'])]
        
        if not consumption_cols:
            return None, None

        consumption = df[consumption_cols[0]].copy()
        
        # Calculate heating consumption: total - baseline
        if 'baseline_consumption' in df.columns:
            heating_consumption = consumption - df['baseline_consumption']
            # Ensure heating consumption is non-negative (can't be negative)
            heating_consumption = heating_consumption.clip(lower=0)
        else:
            # Fallback if baseline not calculated
            heating_consumption = consumption.copy()

        # Select features - exclude heating labels and consumption columns
        exclude_cols = {
            'heating_detected', 'heating_detected_temp', 'heating_detected_pattern', 
            'heating_detected_cluster', 'is_non_heating_period', 'baseline_consumption'
        }
        exclude_cols.update(consumption_cols)  # Exclude original consumption
        
        feature_cols = [col for col in df.columns
                       if col not in exclude_cols
                       and not col.startswith('heating_detected_')]

        if not feature_cols:
            return None, None

        # Create feature matrix
        X = df[feature_cols].copy()
        y = heating_consumption.copy()

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        return X, y

    def _train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models."""
        models = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Linear Regression
        lr_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        lr_model.fit(X_train, y_train)
        models['linear_regression'] = lr_model

        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model

        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model

        return models

    def _evaluate_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models."""
        performance = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for name, model in models.items():
            try:
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Metrics
                metrics = {
                    'train_mse': mean_squared_error(y_train, y_pred_train),
                    'test_mse': mean_squared_error(y_test, y_pred_test),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'cv_score': np.mean(cross_val_score(model, X, y, cv=5, scoring='r2'))
                }

                performance[name] = metrics
                logger.info(f"{name} - Test R²: {metrics['test_r2']:.3f}, CV R²: {metrics['cv_score']:.3f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                performance[name] = {}

        return performance

    def _select_best_model(self, models: Dict[str, Any], performance: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Select the best performing model."""
        best_model = None
        best_score = -np.inf

        for name, metrics in performance.items():
            cv_score = metrics.get('cv_score', -np.inf)
            if cv_score > best_score:
                best_score = cv_score
                best_model = name

        return best_model

    def _save_models(self, models: Dict[str, Any]):
        """Save trained models to disk."""
        for name, model in models.items():
            try:
                model_path = self.models_path / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved model {name} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model {name}: {e}")

    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Make predictions using a trained model."""
        if model_name is None:
            model_name = self.best_model

        if model_name and model_name in self.model_performance:
            model = self.model_performance[model_name]['model']
            return model.predict(X)

        return None