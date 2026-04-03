"""
Preprocessing Agent - Handles data preprocessing and cleaning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

from agents import BaseAgent

logger = logging.getLogger(__name__)


class PreprocessingAgent(BaseAgent):
    """Agent responsible for preprocessing and cleaning data."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("preprocessing_agent", config)

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess the loaded data.

        Args:
            data: Dictionary of DataFrames from DataAgent.

        Returns:
            Dictionary of preprocessed DataFrames.
        """
        processed_data = {}

        for name, df in data.items():
            try:
                processed_df = self._preprocess_dataframe(df, name)
                processed_data[name] = processed_df
                logger.info(f"Preprocessed {name}")
            except Exception as e:
                logger.error(f"Failed to preprocess {name}: {e}")
                processed_data[name] = df  # Return original if preprocessing fails

        return processed_data

    def _preprocess_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Preprocess a single DataFrame."""
        df = df.copy()

        # Handle datetime columns
        if 'timestamp' in df.columns or 'date' in df.columns:
            df = self._process_datetime_columns(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove duplicates
        df = df.drop_duplicates()

        # Specific preprocessing based on data type
        if 'consumption' in name.lower():
            df = self._preprocess_consumption_data(df)
        elif 'airtemp' in name.lower():
            df = self._preprocess_temperature_data(df)

        return df

    def _process_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process datetime columns."""
        for col in df.columns:
            if 'timestamp' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col) if df.index.name is None else df
                except Exception as e:
                    logger.warning(f"Could not parse datetime column {col}: {e}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # For time series data, use forward fill, then backward fill
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.ffill().bfill()
        else:
            # For numeric columns, use interpolation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

        return df

    def _preprocess_consumption_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific preprocessing for consumption data."""
        # Ensure consumption values are positive
        consumption_cols = [col for col in df.columns if 'consumption' in col.lower() or 'energy' in col.lower()]
        for col in consumption_cols:
            df[col] = df[col].clip(lower=0)

        return df

    def _preprocess_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Specific preprocessing for temperature data."""
        # Temperature should be in reasonable range (-50 to 50 Celsius)
        temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'air' in col.lower()]
        for col in temp_cols:
            df[col] = df[col].clip(-50, 50)

        return df