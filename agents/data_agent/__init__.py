"""
Data Agent - Handles data loading and management.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from agents import BaseAgent

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Agent responsible for loading and managing data."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("data_agent", config)
        self.data_path = Path(self.config.get("data_path", "data"))
        # Set data resolution preference: 'hourly', 'quarterly' (15-min), or 'two_minutes'
        self.preferred_resolution = self.config.get("preferred_resolution", "quarterly")
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def run(self, data_files: Optional[List[str]] = None, resolution: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load specified data files or all available data.

        Args:
            data_files: List of data file names to load. If None, loads based on resolution.
            resolution: Preferred resolution ('hourly', 'quarterly', 'two_minutes'). 
                       If specified, loads only that resolution + temperature data.

        Returns:
            Dictionary mapping file names to DataFrames.
        """
        # Determine which files to load
        if resolution:
            if resolution == 'hourly':
                data_files = ["consumption_hourly.csv", "smhi-opendata_1_74460_airtemp.csv"]
            elif resolution == 'quarterly':
                data_files = ["consumption_quarterly.csv", "smhi-opendata_1_74460_airtemp.csv"]
            elif resolution == 'two_minutes':
                data_files = ["consumption_two_minutes.csv", "smhi-opendata_1_74460_airtemp.csv"]
            else:
                logger.warning(f"Unknown resolution {resolution}, loading all data")
                data_files = None
        
        if data_files is None:
            data_files = [
                "consumption_hourly.csv",
                "consumption_quarterly.csv",
                "consumption_two_minutes.csv",
                "smhi-opendata_1_74460_airtemp.csv"
            ]

        loaded_data = {}
        for file_name in data_files:
            if file_name in self._data_cache:
                loaded_data[file_name] = self._data_cache[file_name]
            else:
                try:
                    df = self._load_data_file(file_name)
                    self._data_cache[file_name] = df
                    loaded_data[file_name] = df
                    logger.info(f"Loaded data from {file_name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_name}: {e}")

        return loaded_data

    def _load_data_file(self, file_name: str) -> pd.DataFrame:
        """Load a single data file."""
        file_path = self.data_path / file_name

        if file_name.endswith('.csv'):
            if 'airtemp' in file_name:
                # Special handling for SMHI temperature data
                df = self._load_temperature_data(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

            # Standardize datetime index to be timezone-naive
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    # Convert to UTC and remove timezone info
                    df.index = df.index.tz_convert('UTC').tz_localize(None)

            return df
        else:
            raise ValueError(f"Unsupported file format: {file_name}")

    def _load_temperature_data(self, file_path: Path) -> pd.DataFrame:
        """Load SMHI temperature data with special header handling."""
        # Read the file, skipping metadata lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the actual data header (should contain "Datum;Tid")
        header_line_idx = None
        for i, line in enumerate(lines):
            if 'Datum;Tid' in line:
                header_line_idx = i
                break

        if header_line_idx is None:
            raise ValueError("Could not find data header in temperature file")

        # Read from the header line onwards
        df = pd.read_csv(file_path, sep=';', skiprows=header_line_idx)

        # Combine date and time columns
        if 'Datum' in df.columns and 'Tid (UTC)' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Datum'] + ' ' + df['Tid (UTC)'], utc=True)
            df = df.set_index('timestamp')

        # Clean column names and keep only temperature
        df.columns = [col.split(';')[0] if ';' in col else col for col in df.columns]
        if 'Lufttemperatur' in df.columns:
            df = df[['Lufttemperatur']]
            df.columns = ['temperature']
        else:
            # Try to find temperature column
            temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'luft' in col.lower()]
            if temp_cols:
                df = df[[temp_cols[0]]]
                df.columns = ['temperature']

        # Convert to numeric, handling any non-numeric values
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')

        # Ensure timezone-naive index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    def get_data(self, file_name: str) -> Optional[pd.DataFrame]:
        """Get cached data by file name."""
        return self._data_cache.get(file_name)

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        logger.info("Data cache cleared")