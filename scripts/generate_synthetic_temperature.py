"""
Generate synthetic temperature data for the consumption data period (Jul-Nov 2025).
This fills the gap where real SMHI data is unavailable.

Realistic temperatures for Jönköping, Sweden:
- Summer (Jul-Aug): 15-25°C
- Fall (Sep-Nov): 5-15°C with gradual cooling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Consumption data range
start_date = pd.Timestamp('2025-07-31 22:00:00')
end_date = pd.Timestamp('2025-11-15 12:15:00')

# Generate 15-minute timestamps to match consumption data frequency
dates = pd.date_range(start=start_date, end=end_date, freq='15min')

# Create realistic temperature curve
# Base temperature trend
months = (dates - dates[0]).days / 30.4  # Months from start
base_temp = 20 - 10 * (months / 3.5)  # Cool from 20°C in July to 7°C in Nov

# Add daily cycle (higher during day, lower at night)
hour_of_day = dates.hour
daily_cycle = 5 * np.sin((hour_of_day - 6) * np.pi / 12)  # Peak at 6pm, min at 6am

# Add random weather variation
weather = np.random.normal(0, 2, len(dates))

# Combine
temperature = base_temp + daily_cycle + weather

# Make sure it's realistic
temperature = np.clip(temperature, -5, 30)

# Create DataFrame
df_temp = pd.DataFrame({
    'temperature': temperature
}, index=dates)

df_temp.index.name = 'timestamp'

# Save to CSV
output_file = 'data/temperature_synthetic_2025_07_11.csv'
df_temp.to_csv(output_file)

print(f"Generated synthetic temperature data: {output_file}")
print(f"  Samples: {len(df_temp)}")
print(f"  Range: {df_temp.index.min()} to {df_temp.index.max()}")
print(f"  Temperature stats:")
print(f"    Mean: {df_temp['temperature'].mean():.2f}°C")
print(f"    Min: {df_temp['temperature'].min():.2f}°C")
print(f"    Max: {df_temp['temperature'].max():.2f}°C")
print(f"    Std: {df_temp['temperature'].std():.2f}°C")
print(f"\nFirst few rows:\n{df_temp.head(10)}")
print(f"\nLast few rows:\n{df_temp.tail(10)}")
