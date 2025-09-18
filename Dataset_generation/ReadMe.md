# Dataset Generation

This module provides comprehensive tools for generating synthetic datasets with known causal structures for evaluating causal discovery algorithms. It supports multiple data generation methods and noise types to create controlled experimental conditions.

## Overview

The dataset generation module enables researchers to:
- Generate synthetic datasets with known ground truth DAGs
- Create datasets with different noise characteristics (Gaussian, non-Gaussian)
- Produce linear and non-linear causal relationships
- Generate datasets of varying sizes and complexities
- Access real-world benchmark datasets from public repositories

## Files Description

### Core Generation Functions

- **`data_gen_functions.py`**: Main data generation functions for different causal models
- **`config_datasets.py`**: Configuration parameters for dataset generation
- **`data2.py`**: Utilities for accessing public datasets (CDT, bnlearn)
- **`gaussian_data.py`**: Specialized Gaussian data generation functions

### Demo and Documentation

- **`run.ipynb`**: Jupyter notebook demonstrating dataset generation
- **`requirements.txt`**: Python dependencies for this module
- **`ReadMe.md`**: Basic module description

## Data Generation Methods

### 1. Linear Causal Models

#### Method: `gen_data_given_model()`
- **Description**: Generates data from linear structural equation models
- **Formula**: `X = BX + C + ε` where B is the adjacency matrix
- **Noise Types**: Gaussian, LiNGAM (non-Gaussian)
- **Features**: 
  - Upper triangular adjacency matrix (DAG structure)
  - Configurable edge probabilities
  - Weighted edges with positive/negative coefficients

#### Configuration
```python
config = {
    "method_linear": {
        "node_counts": [10, 40, 70, 100],
        "num_samples": 5000,
        "noise_types": ["lingam", "gaussian"],
        "permutate": True
    }
}
```

### 2. Non-Linear Causal Models

#### Method: `gen_data_given_model_2nd_order()`
- **Description**: Generates data with quadratic (2nd order) causal relationships
- **Features**:
  - Polynomial feature expansion
  - Non-linear parent-child relationships
  - Configurable coefficient sampling
  - NaN handling with mean imputation

#### Configuration
```python
config = {
    "method_2nd_order": {
        "node_counts": [10, 40, 70, 100],
        "num_samples": 1000,
        "noise_types": ["lingam", "gaussian"],
        "permutate": True
    }
}
```

### 3. Gaussian Process Models

#### Method: `generate_datasets_gp()`
- **Description**: Generates data using Gaussian Process causal relationships
- **Features**:
  - RBF kernel-based relationships
  - Non-parametric causal functions
  - Configurable noise variance
  - Topological ordering for DAG generation

#### Configuration
```python
config = {
    "method_gp": {
        "node_counts": [10],
        "num_samples": 1000,
        "noise_variance_range": (0.5, 1.5)
    }
}
```

## Public Dataset Access

### CDT (Causal Discovery Toolbox) Datasets
- **DREAM4**: dream4-1, dream4-2, dream4-3, dream4-4, dream4-5
- **Sachs**: Protein signaling network
- **Access**: `cdt.data.load_dataset(dataset_name)`

### bnlearn Datasets
- **Asia**: Medical diagnosis network
- **Andes**: Student performance network  
- **Alarm**: Medical monitoring network
- **Access**: `bn.bnlearn.import_example(dataset_name)`

## Usage Examples

### Basic Linear Data Generation

```python
from data_gen_functions import generate_datasets
from config_datasets import config

# Generate linear datasets
datasets, adj_matrices = generate_datasets(
    node_counts=[10, 20, 50],
    edge_probability=0.3,
    n_samples=1000,
    noise_type='gaussian',
    permutate=True
)
```

### Non-Linear Data Generation

```python
from data_gen_functions import generate_datasets_quad

# Generate quadratic datasets
datasets, adj_matrices = generate_datasets_quad(
    node_counts=[10, 20],
    edge_probability=0.4,
    n_samples=500,
    noise_type='lingam',
    permutate=True
)
```

### Gaussian Process Data Generation

```python
from data_gen_functions import generate_datasets_gp

# Generate GP datasets
datasets, adj_matrices = generate_datasets_gp(
    node_counts=[10, 15],
    edge_probability=0.3,
    num_samples=800,
    noise_variance_range=(0.5, 1.5)
)
```

### Accessing Public Datasets

```python
from data2 import cdt_data, bnlearn_data

# Load DREAM4 datasets
dream4_datasets = cdt_data.dream4_all()

# Load individual datasets
sachs_data = cdt_data.sachs()
asia_data = bnlearn_data.asia()
alarm_data = bnlearn_data.alarm()
```

## Configuration Parameters

### Common Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `node_counts` | Number of variables | [10, 40, 100] | List of integers |
| `edge_probability` | Edge formation probability | 0.5 | 0.0 - 1.0 |
| `num_samples` | Number of observations | 5000 | Positive integer |
| `noise_types` | Noise distribution | ["gaussian"] | ["gaussian", "lingam"] |
| `permutate` | Randomize variable order | True | Boolean |

### Noise Types

#### Gaussian Noise
- **Distribution**: Standard normal N(0,1)
- **Characteristics**: Symmetric, light-tailed
- **Use Case**: Standard linear models

#### LiNGAM Noise
- **Distribution**: Non-Gaussian, heavy-tailed
- **Characteristics**: Asymmetric, sparse
- **Use Case**: Independent Component Analysis

### Edge Weight Ranges

- **Positive Weights**: 0.5 to 2.0
- **Negative Weights**: -2.0 to -0.5
- **Sampling**: Uniform distribution within ranges

## Output Files

### Generated Files

For each dataset configuration, the following files are generated:

1. **Data Files**:
   - `{noise_type}_{nodes}nodes.csv`: Dataset observations
   - `{noise_type}_{nodes}nodes_non-linear_quad.csv`: Quadratic datasets
   - `gaussian_{nodes}nodes_GP.csv`: Gaussian Process datasets

2. **Adjacency Matrices**:
   - `adj_matrix_{nodes}nodes_linear.csv`: Linear model adjacency
   - `adj_matrix_{nodes}nodes_non-linear_quad.csv`: Quadratic model adjacency
   - `adj_matrix_{nodes}nodes_GP.csv`: GP model adjacency

### File Formats

- **CSV Format**: Comma-separated values with headers
- **Headers**: `Feature 0`, `Feature 1`, ..., `Feature N-1`
- **Data Type**: Float64 for numerical precision
- **Missing Values**: Handled with mean imputation

## Advanced Features

### Graph Generation

```python
from data_gen_functions import generate_W, generate_graph_with_edges

# Generate weighted adjacency matrix
W = generate_W(d=10, prob=0.3)

# Generate NetworkX DAG
G = generate_graph_with_edges(d=10, prob=0.3)
```

### Data Preprocessing

```python
from data_gen_functions import preprocess_adj_matrix, replace_nan_with_mean

# Preprocess adjacency matrix
binary_adj = preprocess_adj_matrix(adj_matrix, threshold=0.5)

# Handle missing values
clean_data = replace_nan_with_mean(data)
```

### Custom Configurations

```python
# Custom configuration
custom_config = {
    "method_linear": {
        "node_counts": [5, 15, 25],
        "num_samples": 2000,
        "noise_types": ["gaussian"],
        "permutate": False
    }
}
```

## Performance Considerations

### Memory Usage
- **Small datasets** (< 1000 samples): Minimal memory usage
- **Medium datasets** (1000-10000 samples): Moderate memory usage
- **Large datasets** (> 10000 samples): High memory usage

### Generation Time
- **Linear models**: Fast generation (seconds)
- **Quadratic models**: Moderate generation (minutes)
- **GP models**: Slow generation (minutes to hours)

### Optimization Tips
1. Use appropriate sample sizes for your use case
2. Generate datasets in batches for large experiments
3. Save intermediate results to avoid regeneration
4. Use parallel processing for multiple configurations

## Validation and Quality Control

### DAG Validation

```python
import networkx as nx

# Verify DAG property
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
is_dag = nx.is_directed_acyclic_graph(G)
print(f"Is DAG: {is_dag}")
```

### Data Quality Checks

```python
# Check for missing values
missing_count = data.isnull().sum().sum()
print(f"Missing values: {missing_count}")

# Check data statistics
print(data.describe())

# Check for infinite values
inf_count = np.isinf(data).sum().sum()
print(f"Infinite values: {inf_count}")
```

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `networkx`: Graph operations
- `scikit-learn`: Polynomial features and preprocessing
- `GPy`: Gaussian Process modeling
- `cdt`: Causal Discovery Toolbox
- `bnlearn`: Bayesian network learning

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce sample size or node count
2. **Convergence Issues**: Check noise parameters and edge probabilities
3. **DAG Violations**: Verify adjacency matrix generation
4. **Import Errors**: Install missing dependencies

### Performance Issues

1. **Slow Generation**: Use smaller datasets or simpler models
2. **High Memory Usage**: Process datasets in chunks
3. **Convergence Problems**: Adjust noise parameters

## Contributing

When adding new generation methods:
1. Follow the existing function naming conventions
2. Add configuration parameters to `config_datasets.py`
3. Include comprehensive docstrings
4. Add validation functions
5. Update this documentation

## Examples

See `run.ipynb` for complete examples of:
- Generating datasets with different parameters
- Accessing public benchmark datasets
- Validating generated data
- Comparing different generation methods
- Exporting results for analysis