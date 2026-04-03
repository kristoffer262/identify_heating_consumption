# identify_heating_consumption

Project aim: Identify and quantify what part of household energy consumption is heating-related.

## Approach

1. **Heating Detection**: Use temperature as primary indicator to identify periods when heating is likely active (temperature < 18°C)
2. **Baseline Estimation**: Calculate baseline (non-heating) consumption during warm periods
3. **Heating Prediction**: Train regression model to predict heating consumption component during cold periods
4. **Decomposition**: Separate total consumption into heating and non-heating components

## Key Metrics

- Heating consumption (kWh) - predicted heating energy use
- Non-heating consumption (kWh) - baseline energy use
- Heating percentage (%) - proportion of total consumption from heating
