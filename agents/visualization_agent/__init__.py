"""
Visualization Agent - Creates visualizations for data analysis and results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from agents import BaseAgent

logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """Agent responsible for creating visualizations."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("visualization_agent", config)
        self.plots_path = Path(self.config.get("plots_path", "plots"))
        self.plots_path.mkdir(exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

    def run(self, data: Dict[str, pd.DataFrame], evaluation_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create visualizations for the analysis results.

        Args:
            data: Dictionary of processed DataFrames.
            evaluation_results: Results from EvaluationAgent.

        Returns:
            Dictionary mapping plot names to file paths.
        """
        plot_files = {}

        try:
            # Data exploration plots
            plot_files.update(self._create_data_exploration_plots(data))

            # Heating detection plots
            plot_files.update(self._create_heating_detection_plots(data))

            # Time series plots
            plot_files.update(self._create_time_series_plots(data))

            # Model evaluation plots (if available)
            if evaluation_results:
                plot_files.update(self._create_model_evaluation_plots(evaluation_results))

            # Interactive dashboard
            plot_files.update(self._create_interactive_dashboard(data))

            logger.info(f"Created {len(plot_files)} visualizations")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")

        return plot_files

    def _create_data_exploration_plots(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create data exploration plots."""
        plots = {}

        # Consumption distribution
        consumption_data = self._get_consumption_data(data)
        if consumption_data is not None:
            plots['consumption_distribution'] = self._plot_consumption_distribution(consumption_data)

        # Temperature distribution
        temp_data = self._get_temperature_data(data)
        if temp_data is not None:
            plots['temperature_distribution'] = self._plot_temperature_distribution(temp_data)

        # Correlation heatmap
        plots['correlation_heatmap'] = self._plot_correlation_heatmap(data)

        return plots

    def _create_heating_detection_plots(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create heating detection visualization plots."""
        plots = {}

        consumption_data = self._get_consumption_data(data)
        if consumption_data is not None and 'heating_detected' in consumption_data.columns:
            # Heating periods over time
            plots['heating_periods_timeline'] = self._plot_heating_periods_timeline(consumption_data)

            # Consumption by heating status
            plots['consumption_by_heating'] = self._plot_consumption_by_heating_status(consumption_data)

            # Temperature vs consumption with heating detection
            plots['temp_vs_consumption_heating'] = self._plot_temperature_vs_consumption(consumption_data)

        return plots

    def _create_time_series_plots(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create time series plots."""
        plots = {}

        consumption_data = self._get_consumption_data(data)
        if consumption_data is not None:
            # Daily consumption patterns
            plots['daily_consumption_pattern'] = self._plot_daily_consumption_pattern(consumption_data)

            # Weekly consumption patterns
            plots['weekly_consumption_pattern'] = self._plot_weekly_consumption_pattern(consumption_data)

            # Seasonal consumption
            plots['seasonal_consumption'] = self._plot_seasonal_consumption(consumption_data)

        return plots

    def _create_model_evaluation_plots(self, evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """Create model evaluation plots."""
        plots = {}

        # Model comparison (reuse from evaluation agent if available)
        if 'model_comparison' in evaluation_results:
            comparison_df = evaluation_results['model_comparison']
            plots['model_comparison'] = self._plot_model_comparison(comparison_df)

        return plots

    def _create_interactive_dashboard(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create interactive dashboard."""
        plots = {}

        try:
            consumption_data = self._get_consumption_data(data)
            temp_data = self._get_temperature_data(data)

            if consumption_data is not None:
                plots['interactive_dashboard'] = self._create_plotly_dashboard(consumption_data, temp_data)

        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")

        return plots

    def _get_consumption_data(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Get consumption data from the data dictionary."""
        for name, df in data.items():
            if 'consumption' in name.lower():
                return df
        return None

    def _get_temperature_data(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Get temperature data from the data dictionary."""
        for name, df in data.items():
            if 'airtemp' in name.lower():
                return df
        return None

    def _plot_consumption_distribution(self, df: pd.DataFrame) -> str:
        """Plot consumption distribution."""
        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]

        if not consumption_cols:
            return ""

        fig, axes = plt.subplots(1, len(consumption_cols), figsize=(6*len(consumption_cols), 5))

        if len(consumption_cols) == 1:
            axes = [axes]

        for i, col in enumerate(consumption_cols):
            sns.histplot(df[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('Consumption')
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        file_path = f"{self.plots_path}/consumption_distribution.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_temperature_distribution(self, df: pd.DataFrame) -> str:
        """Plot temperature distribution."""
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]

        if not temp_cols:
            return ""

        plt.figure(figsize=(8, 6))
        for col in temp_cols:
            sns.histplot(df[col], kde=True, label=col)

        plt.title('Temperature Distribution')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Frequency')
        plt.legend()

        file_path = f"{self.plots_path}/temperature_distribution.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_correlation_heatmap(self, data: Dict[str, pd.DataFrame]) -> str:
        """Plot correlation heatmap."""
        # Combine all numeric data
        all_data = []
        for name, df in data.items():
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                # Rename columns to include source
                numeric_df.columns = [f"{name}_{col}" for col in numeric_df.columns]
                all_data.append(numeric_df)

        if not all_data:
            return ""

        combined_df = pd.concat(all_data, axis=1)

        # Calculate correlation
        corr_matrix = combined_df.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Heatmap')

        file_path = f"{self.plots_path}/correlation_heatmap.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_heating_periods_timeline(self, df: pd.DataFrame) -> str:
        """Plot heating periods over time."""
        if 'heating_detected' not in df.columns:
            return ""

        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot consumption
        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]
        if consumption_cols:
            ax.plot(df.index, df[consumption_cols[0]], alpha=0.7, label='Consumption')

        # Highlight heating periods
        heating_periods = df[df['heating_detected'] == 1]
        if not heating_periods.empty:
            ax.fill_between(heating_periods.index,
                          heating_periods[consumption_cols[0]] if consumption_cols else 0,
                          alpha=0.3, color='red', label='Heating Period')

        ax.set_title('Consumption with Heating Periods')
        ax.set_xlabel('Time')
        ax.set_ylabel('Consumption')
        ax.legend()

        file_path = f"{self.plots_path}/heating_periods_timeline.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_consumption_by_heating_status(self, df: pd.DataFrame) -> str:
        """Plot consumption distribution by heating status."""
        if 'heating_detected' not in df.columns:
            return ""

        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]

        if not consumption_cols:
            return ""

        fig, axes = plt.subplots(1, len(consumption_cols), figsize=(6*len(consumption_cols), 5))

        if len(consumption_cols) == 1:
            axes = [axes]

        for i, col in enumerate(consumption_cols):
            sns.boxplot(x='heating_detected', y=col, data=df, ax=axes[i])
            axes[i].set_title(f'{col} by Heating Status')
            axes[i].set_xlabel('Heating Detected (0=No, 1=Yes)')

        plt.tight_layout()
        file_path = f"{self.plots_path}/consumption_by_heating.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_temperature_vs_consumption(self, df: pd.DataFrame) -> str:
        """Plot temperature vs consumption with heating detection."""
        temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]

        if not temp_cols or not consumption_cols or 'heating_detected' not in df.columns:
            return ""

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[temp_cols[0]], y=df[consumption_cols[0]],
                       hue=df['heating_detected'], palette='coolwarm', alpha=0.6)
        plt.title('Temperature vs Consumption (Heating Detection)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Consumption')

        file_path = f"{self.plots_path}/temp_vs_consumption_heating.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_daily_consumption_pattern(self, df: pd.DataFrame) -> str:
        """Plot daily consumption patterns."""
        if not isinstance(df.index, pd.DatetimeIndex) or 'hour' not in df.columns:
            return ""

        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]

        if not consumption_cols:
            return ""

        # Group by hour of day
        hourly_avg = df.groupby('hour')[consumption_cols[0]].mean()

        plt.figure(figsize=(10, 6))
        hourly_avg.plot(kind='line', marker='o')
        plt.title('Average Consumption by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Average Consumption')
        plt.grid(True, alpha=0.3)

        file_path = f"{self.plots_path}/daily_consumption_pattern.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_weekly_consumption_pattern(self, df: pd.DataFrame) -> str:
        """Plot weekly consumption patterns."""
        if not isinstance(df.index, pd.DatetimeIndex) or 'day_of_week' not in df.columns:
            return ""

        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]

        if not consumption_cols:
            return ""

        # Group by day of week
        weekly_avg = df.groupby('day_of_week')[consumption_cols[0]].mean()

        plt.figure(figsize=(10, 6))
        weekly_avg.plot(kind='bar')
        plt.title('Average Consumption by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Average Consumption')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        file_path = f"{self.plots_path}/weekly_consumption_pattern.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_seasonal_consumption(self, df: pd.DataFrame) -> str:
        """Plot seasonal consumption patterns."""
        if not isinstance(df.index, pd.DatetimeIndex) or 'month' not in df.columns:
            return ""

        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() and not 'rolling' in col]

        if not consumption_cols:
            return ""

        # Group by month
        monthly_avg = df.groupby('month')[consumption_cols[0]].mean()

        plt.figure(figsize=(10, 6))
        monthly_avg.plot(kind='bar')
        plt.title('Average Consumption by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Consumption')
        plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        file_path = f"{self.plots_path}/seasonal_consumption.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> str:
        """Plot model comparison."""
        if comparison_df.empty or 'test_r2' not in comparison_df.columns:
            return ""

        plt.figure(figsize=(10, 6))
        comparison_df.plot(x='model', y='test_r2', kind='bar')
        plt.title('Model Performance Comparison (Test R²)')
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)

        file_path = f"{self.plots_path}/model_comparison.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        return file_path

    def _create_plotly_dashboard(self, consumption_df: pd.DataFrame, temp_df: Optional[pd.DataFrame]) -> str:
        """Create interactive Plotly dashboard."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Consumption Over Time', 'Temperature Over Time',
                              'Consumption Distribution', 'Heating Detection'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                      [{'type': 'histogram'}, {'type': 'scatter'}]]
            )

            # Consumption over time
            consumption_cols = [col for col in consumption_df.columns if 'consumption' in col.lower() and not 'rolling' in col]
            if consumption_cols:
                fig.add_trace(
                    go.Scatter(x=consumption_df.index, y=consumption_df[consumption_cols[0]],
                             mode='lines', name='Consumption'),
                    row=1, col=1
                )

            # Temperature over time
            if temp_df is not None:
                temp_cols = [col for col in temp_df.columns if 'temp' in col.lower()]
                if temp_cols:
                    fig.add_trace(
                        go.Scatter(x=temp_df.index, y=temp_df[temp_cols[0]],
                                 mode='lines', name='Temperature', line=dict(color='red')),
                        row=1, col=2
                    )

            # Consumption distribution
            if consumption_cols:
                fig.add_trace(
                    go.Histogram(x=consumption_df[consumption_cols[0]], name='Consumption'),
                    row=2, col=1
                )

            # Heating detection scatter
            if 'heating_detected' in consumption_df.columns and consumption_cols:
                colors = ['blue' if h == 0 else 'red' for h in consumption_df['heating_detected']]
                fig.add_trace(
                    go.Scatter(x=consumption_df.index, y=consumption_df[consumption_cols[0]],
                             mode='markers', name='Heating Detection',
                             marker=dict(color=colors, size=4)),
                    row=2, col=2
                )

            fig.update_layout(height=800, title_text="Heating Consumption Analysis Dashboard")
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Consumption", row=1, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            # Save as HTML
            file_path = f"{self.plots_path}/interactive_dashboard.html"
            fig.write_html(file_path)

            return file_path

        except Exception as e:
            logger.error(f"Failed to create Plotly dashboard: {e}")
            return ""