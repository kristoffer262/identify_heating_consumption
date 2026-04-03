"""
Feature Agent - Creates features for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from agents import BaseAgent

logger = logging.getLogger(__name__)


class FeatureAgent(BaseAgent):
    """Agent responsible for creating features for modeling."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("feature_agent", config)

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create features from preprocessed data.

        Args:
            data: Dictionary of preprocessed DataFrames.

        Returns:
            Dictionary of DataFrames with additional features.
        """
        featured_data = {}

        # Get consumption and temperature data
        consumption_data = self._get_consumption_data(data)
        temperature_data = self._get_temperature_data(data)

        for name, df in data.items():
            try:
                featured_df = self._create_features(df, name, temperature_data)
                featured_data[name] = featured_df
                logger.info(f"Created features for {name}")
            except Exception as e:
                logger.error(f"Failed to create features for {name}: {e}")
                featured_data[name] = df

        return featured_data

    def _get_consumption_data(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Get the main consumption data."""
        for name, df in data.items():
            if 'consumption' in name.lower() and 'hourly' in name.lower():
                return df
        return None

    def _get_temperature_data(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Get the temperature data."""
        for name, df in data.items():
            if 'temperature' in name.lower() or 'airtemp' in name.lower():
                return df
        return None

    def _create_features(self, df: pd.DataFrame, name: str,
                        temperature_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Create features for a DataFrame."""
        df = df.copy()

        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df = self._create_time_features(df)

        # Temperature features
        if temperature_data is not None and 'consumption' in name.lower():
            df = self._create_temperature_features(df, temperature_data)

        # Statistical features
        df = self._create_statistical_features(df)

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()

        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

        # Seasonal features
        df['is_heating_season'] = ((df.index.month >= 10) | (df.index.month <= 4)).astype(int)

        # Time of day categories
        df['time_of_day'] = pd.cut(df.index.hour,
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  right=False)

        return df

    def _create_temperature_features(self, df: pd.DataFrame,
                                   temperature_data: pd.DataFrame) -> pd.DataFrame:
        """Create temperature-related features."""
        df = df.copy()

        # Align temperature data with consumption data
        try:
            if not df.index.equals(temperature_data.index):
                # Use join to align indices, forward fill any gaps
                temp_aligned = temperature_data.reindex(df.index, fill_value=np.nan)
                temp_aligned = temp_aligned.ffill().bfill()
            else:
                temp_aligned = temperature_data

            # Add temperature features
            temp_col = [col for col in temp_aligned.columns if 'temp' in col.lower()]
            if temp_col and len(temp_aligned) > 0 and temp_aligned[temp_col[0]].notna().sum() > 0:
                df['temperature'] = temp_aligned[temp_col[0]].values

                # Temperature categories
                df['temp_category'] = pd.cut(df['temperature'],
                                           bins=[-50, 0, 10, 20, 50],
                                           labels=['freezing', 'cold', 'mild', 'warm'])

                # Heating degree days (simplified)
                df['heating_degree_days'] = np.maximum(18 - df['temperature'], 0)
        except Exception as e:
            logger.warning(f"Failed to create temperature features: {e}")

        return df

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        df = df.copy()

        # Rolling statistics for consumption data
        consumption_cols = [col for col in df.columns if 'consumption' in col.lower()]

        for col in consumption_cols:
            # Rolling means
            df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_rolling_mean_7d'] = df[col].rolling(window=168, min_periods=1).mean()

            # Rolling standard deviations
            df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24, min_periods=1).std()

        return df