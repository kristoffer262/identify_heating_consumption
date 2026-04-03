"""
Evaluation Agent - Evaluates model performance and provides insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve

from agents import BaseAgent

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """Agent responsible for evaluating model performance."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("evaluation_agent", config)
        self.plots_path = self.config.get("plots_path", "plots")
        Path(self.plots_path).mkdir(exist_ok=True)

    def run(self, modeling_results: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate model performance and generate insights.

        Args:
            modeling_results: Results from ModelingAgent.
            data: Original processed data.

        Returns:
            Dictionary containing evaluation results and insights.
        """
        evaluation_results = {}

        try:
            models = modeling_results.get('models', {})
            performance = modeling_results.get('performance', {})

            if not models:
                logger.error("No models found for evaluation")
                return evaluation_results

            # Comprehensive evaluation
            evaluation_results['model_comparison'] = self._compare_models(performance)
            evaluation_results['detailed_metrics'] = self._calculate_detailed_metrics(models, data)
            evaluation_results['feature_importance'] = self._analyze_feature_importance(models)
            evaluation_results['model_insights'] = self._generate_model_insights(models, data)

            # Generate plots
            self._generate_evaluation_plots(models, data, performance, modeling_results)

            logger.info("Evaluation completed successfully")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

        return evaluation_results

    def _compare_models(self, performance: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare model performance across metrics."""
        comparison_data = []

        for model_name, metrics in performance.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def _calculate_detailed_metrics(self, models: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Calculate detailed evaluation metrics."""
        detailed_metrics = {}

        # Find consumption data
        consumption_data = None
        for name, df in data.items():
            if 'consumption' in name.lower() and 'heating_detected' in df.columns:
                consumption_data = df
                break

        if consumption_data is None:
            return detailed_metrics

        # Prepare data (similar to modeling agent)
        from agents.modeling_agent import ModelingAgent
        modeling_agent = ModelingAgent()
        X, y = modeling_agent._prepare_modeling_data(consumption_data)

        if X is None or y is None:
            return detailed_metrics

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)

                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                    'max_error': np.max(np.abs(y_test - y_pred))
                }

                detailed_metrics[model_name] = metrics

            except Exception as e:
                logger.error(f"Failed to calculate metrics for {model_name}: {e}")
                detailed_metrics[model_name] = {}

        return detailed_metrics

    def _analyze_feature_importance(self, models: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Analyze feature importance for tree-based models."""
        feature_importance = {}

        for model_name, model in models.items():
            try:
                # Check if model has feature_importances_ attribute
                if hasattr(model, 'feature_importances_'):
                    # For pipeline models, get the actual estimator
                    if hasattr(model, 'named_steps'):
                        estimator = model.named_steps.get('regressor', model)
                        if hasattr(estimator, 'feature_importances_'):
                            importance = estimator.feature_importances_
                        else:
                            continue
                    else:
                        importance = model.feature_importances_

                    # Get feature names (this is approximate)
                    if hasattr(model, 'n_features_in_'):
                        n_features = model.n_features_in_
                        feature_names = [f'feature_{i}' for i in range(n_features)]
                    else:
                        feature_names = [f'feature_{i}' for i in range(len(importance))]

                    importance_series = pd.Series(importance, index=feature_names)
                    importance_series = importance_series.sort_values(ascending=False)

                    feature_importance[model_name] = importance_series

            except Exception as e:
                logger.error(f"Failed to analyze feature importance for {model_name}: {e}")

        return feature_importance

    def _generate_model_insights(self, models: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate insights about model performance and data."""
        insights = {}

        # Find best performing model
        performance_comparison = self._compare_models({})
        if not performance_comparison.empty:
            best_model = performance_comparison.loc[performance_comparison['test_r2'].idxmax(), 'model']
            insights['best_model'] = best_model

        # Data insights
        consumption_data = None
        for name, df in data.items():
            if 'consumption' in name.lower():
                consumption_data = df
                break

        if consumption_data is not None:
            insights['data_stats'] = {
                'total_samples': len(consumption_data),
                'date_range': f"{consumption_data.index.min()} to {consumption_data.index.max()}" if isinstance(consumption_data.index, pd.DatetimeIndex) else None,
                'heating_periods': consumption_data.get('heating_detected', 0).sum() if 'heating_detected' in consumption_data.columns else 0
            }

        return insights

    def _generate_evaluation_plots(self, models: Dict[str, Any], data: Dict[str, pd.DataFrame],
                                 performance: Dict[str, Dict[str, float]], modeling_results: Dict[str, Any]):
        """Generate evaluation plots."""
        try:
            # Model comparison plot
            self._plot_model_comparison(performance)

            # Feature importance plot (if available)
            feature_importance = self._analyze_feature_importance(models)
            if feature_importance:
                self._plot_feature_importance(feature_importance)

            # Prediction vs actual plot for best model
            self._plot_predictions_vs_actual(models, data)

            # Test set consumption and heating breakdown plot
            test_set = modeling_results.get('test_set')
            if isinstance(test_set, pd.DataFrame) and not test_set.empty:
                self._plot_test_consumption_vs_heating(test_set)

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

    def _plot_model_comparison(self, performance: Dict[str, Dict[str, float]]):
        """Plot model comparison."""
        comparison_df = self._compare_models(performance)

        if comparison_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison')

        # R² scores
        if 'test_r2' in comparison_df.columns:
            comparison_df.plot(x='model', y='test_r2', kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Test R² Score')
            axes[0,0].tick_params(axis='x', rotation=45)

        # MSE scores
        if 'test_mse' in comparison_df.columns:
            comparison_df.plot(x='model', y='test_mse', kind='bar', ax=axes[0,1])
            axes[0,1].set_title('Test MSE')
            axes[0,1].tick_params(axis='x', rotation=45)

        # CV scores
        if 'cv_score' in comparison_df.columns:
            comparison_df.plot(x='model', y='cv_score', kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Cross-Validation R²')
            axes[1,0].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.plots_path}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, feature_importance: Dict[str, pd.Series]):
        """Plot feature importance."""
        for model_name, importance in feature_importance.items():
            plt.figure(figsize=(10, 6))
            importance.head(20).plot(kind='barh')
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f"{self.plots_path}/feature_importance_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_predictions_vs_actual(self, models: Dict[str, Any], data: Dict[str, pd.DataFrame]):
        """Plot predictions vs actual values."""
        # Find consumption data
        consumption_data = None
        for name, df in data.items():
            if 'consumption' in name.lower() and 'heating_detected' in df.columns:
                consumption_data = df
                break

        if consumption_data is None:
            return

        # Prepare data
        from agents.modeling_agent import ModelingAgent
        modeling_agent = ModelingAgent()
        X, y = modeling_agent._prepare_modeling_data(consumption_data)

        if X is None or y is None:
            return

        # Use first model for plotting
        model_name = list(models.keys())[0]
        model = models[model_name]

        try:
            y_pred = model.predict(X)

            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Actual Consumption')
            plt.ylabel('Predicted Consumption')
            plt.title(f'Predictions vs Actual - {model_name}')
            plt.tight_layout()
            plt.savefig(f"{self.plots_path}/predictions_vs_actual_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to plot predictions vs actual: {e}")

    def _plot_test_consumption_vs_heating(self, test_set: pd.DataFrame):
        """Plot test set total consumption and heating component."""
        if test_set.empty:
            return

        if 'total_consumption' not in test_set.columns or 'heating_consumption' not in test_set.columns:
            return

        # Sort by timestamp because train_test_split shuffles the index
        if isinstance(test_set.index, pd.DatetimeIndex):
            test_set = test_set.sort_index()

        plt.figure(figsize=(14, 6))
        plt.plot(test_set.index, test_set['total_consumption'], label='Total Consumption', alpha=0.8, linewidth=1)
        plt.plot(test_set.index, test_set['heating_consumption'], label='Heating Consumption', alpha=0.8, linewidth=1)

        plt.title('Test Set: Total Consumption and Heating Consumption')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.grid(alpha=0.3)

        file_path = f"{self.plots_path}/test_set_total_vs_heating.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path