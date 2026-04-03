"""
Heating Detection Agent - Detects heating consumption periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from agents import BaseAgent

logger = logging.getLogger(__name__)


class HeatingDetectionAgent(BaseAgent):
    """Agent responsible for detecting heating consumption periods."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("heating_detection_agent", config)
        self.heating_threshold = self.config.get("heating_threshold", 18)  # Celsius
        self.min_heating_hours = self.config.get("min_heating_hours", 2)

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Detect heating periods in consumption data.

        Args:
            data: Dictionary of DataFrames with features.

        Returns:
            Dictionary of DataFrames with heating detection labels.
        """
        detected_data = {}

        for name, df in data.items():
            if 'consumption' in name.lower():
                try:
                    detected_df = self._detect_heating_periods(df)
                    detected_data[name] = detected_df
                    logger.info(f"Detected heating periods in {name}")
                except Exception as e:
                    logger.error(f"Failed to detect heating in {name}: {e}")
                    detected_data[name] = df
            else:
                detected_data[name] = df

        return detected_data

    def _detect_heating_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect heating periods in consumption data."""
        df = df.copy()

        # Method 1: Temperature-based detection (if temperature data available)
        if 'temperature' in df.columns and df['temperature'].notna().sum() > 0:
            df['heating_detected_temp'] = self._temperature_based_detection(df)
        else:
            df['heating_detected_temp'] = 0  # No temperature data available

        # Method 2: Consumption pattern-based detection
        df['heating_detected_pattern'] = self._pattern_based_detection(df)

        # Method 3: Clustering-based detection
        df['heating_detected_cluster'] = self._clustering_based_detection(df)

        # Combined detection (majority vote or fallback to pattern-based)
        heating_cols = [col for col in df.columns if col.startswith('heating_detected_')]
        if len(heating_cols) > 1:
            # Use pattern and cluster methods if temperature not available
            active_cols = [col for col in heating_cols if col != 'heating_detected_temp' or df['heating_detected_temp'].sum() > 0]
            df['heating_detected'] = df[active_cols].mean(axis=1).round().astype(int)
        else:
            df['heating_detected'] = df['heating_detected_pattern']

        # Apply minimum duration filter
        df['heating_detected'] = self._filter_minimum_duration(df['heating_detected'])

        return df

    def _temperature_based_detection(self, df: pd.DataFrame) -> pd.Series:
        """Detect heating based on temperature thresholds."""
        return (df['temperature'] < self.heating_threshold).astype(int)

    def _pattern_based_detection(self, df: pd.DataFrame) -> pd.Series:
        """Detect heating based on consumption patterns."""
        consumption_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['consumption', 'energy', 'kwh'])]

        if not consumption_cols:
            return pd.Series(0, index=df.index)

        consumption = df[consumption_cols[0]]

        # Method 1: Seasonal pattern (higher consumption in heating season)
        if 'is_heating_season' in df.columns:
            seasonal_heating = df['is_heating_season'].astype(int)
        else:
            # Fallback: assume heating season is Oct-Mar
            seasonal_heating = ((df.index.month >= 10) | (df.index.month <= 3)).astype(int)

        # Method 2: Daily pattern (higher consumption during cold hours)
        if 'hour' in df.columns:
            # Assume heating is more likely during early morning and evening
            cold_hours = ((df['hour'] >= 6) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 22))
            daily_pattern = cold_hours.astype(int)
        else:
            daily_pattern = pd.Series(0, index=df.index)

        # Method 3: Consumption spikes above baseline
        # Calculate rolling baseline
        baseline = consumption.rolling(window=24*7, min_periods=24).quantile(0.25)  # 25th percentile over week
        baseline = baseline.fillna(consumption.quantile(0.25))  # Fallback
        spike_detection = (consumption > baseline * 1.5).astype(int)

        # Combine methods
        combined_score = (seasonal_heating + daily_pattern + spike_detection) / 3
        return (combined_score > 0.5).astype(int)

    def _clustering_based_detection(self, df: pd.DataFrame) -> pd.Series:
        """Detect heating using unsupervised clustering."""
        # Prepare features for clustering
        features = []

        if 'temperature' in df.columns:
            features.append(df['temperature'])

        consumption_cols = [col for col in df.columns if 'consumption' in col.lower()]
        if consumption_cols:
            features.append(df[consumption_cols[0]])

        if 'hour' in df.columns:
            features.append(df['hour'])

        if not features:
            return pd.Series(0, index=df.index)

        # Create feature matrix
        X = pd.concat(features, axis=1).dropna()

        if len(X) < 10:  # Not enough data for clustering
            return pd.Series(0, index=df.index)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Assume the cluster with higher consumption is heating
        cluster_consumption = pd.Series(clusters, index=X.index).groupby(clusters).apply(
            lambda x: df.loc[x.index, consumption_cols[0]].mean() if consumption_cols else 0
        )

        heating_cluster = cluster_consumption.idxmax()

        # Map back to original index
        result = pd.Series(0, index=df.index)
        result.loc[X.index] = (clusters == heating_cluster).astype(int)

        return result

    def _filter_minimum_duration(self, heating_series: pd.Series) -> pd.Series:
        """Filter heating periods to ensure minimum duration."""
        # Find consecutive heating periods
        groups = (heating_series != heating_series.shift()).cumsum()
        group_sizes = heating_series.groupby(groups).transform('size')

        # Keep only groups that meet minimum duration
        return heating_series.where(group_sizes >= self.min_heating_hours, 0)