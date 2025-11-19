# Datasets

## Overview

4 datasets for Causal SLM evaluation (2 synthetic + 2 real):

### 1. Synthetic HVAC (synthetic_hvac.csv)
- **Type**: Synthetic with known causal structure
- **Variables**: Occupancy, HVAC status, Temperature, Humidity
- **Samples**: 5000
- **Causal structure**: Occupancy → HVAC → Temperature, Humidity
- **Use case**: Ground truth validation

### 2. Synthetic Industrial IoT (synthetic_industrial.csv)
- **Type**: Synthetic with known causal structure
- **Variables**: Load, Vibration, Temperature, Failure
- **Samples**: 5000
- **Causal structure**: Load → Vibration → Temperature → Failure (chain)
- **Use case**: Predictive maintenance validation

### 3. Real HVAC (real_hvac.csv)
- **Type**: Realistic patterns (simulated from real-world)
- **Variables**: Occupancy, Outdoor temp, HVAC, Indoor temp, Humidity, Energy
- **Samples**: 5000
- **Patterns**: Business hours, daily cycles, seasonal trends
- **Use case**: Real-world performance evaluation

### 4. Real Environmental (real_environmental.csv)
- **Type**: Realistic air quality patterns
- **Variables**: Traffic, Wind, Temperature, CO, PM2.5, AQI, Health alert
- **Samples**: 5000
- **Patterns**: Rush hour traffic, wind dispersion
- **Use case**: Environmental monitoring validation

## Files

- `*.csv`: Dataset files (timestamped sensor readings)
- `*_graph.npy`: Ground truth causal graphs (NumPy arrays)
- `README.md`: This file

## Usage

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/synthetic_hvac.csv')

# Load ground truth causal graph
graph = np.load('data/synthetic_hvac_graph.npy')
```

## Evaluation Strategy

1. **Synthetic datasets**: Test causal discovery accuracy (known ground truth)
2. **Real datasets**: Test real-world performance and robustness
3. **Cross-dataset**: Test generalization across domains

Generated: October 28, 2025
