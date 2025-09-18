# Data Directory

This directory contains benchmark datasets used for evaluating causal discovery algorithms in the Causal LLM Framework. The datasets include both real-world and synthetic data with known ground truth causal structures.

## Directory Structure

```
Data/
├── alarm/                    # ALARM network dataset
├── asia/                     # Asia network dataset  
├── BIAS/                     # BIAS dataset with multiple variants
├── dream41/                  # DREAM4 Challenge dataset 1
├── dream42/                  # DREAM4 Challenge dataset 2
├── dream43/                  # DREAM4 Challenge dataset 3
├── dream44/                  # DREAM4 Challenge dataset 4
├── Hepar2/                   # Hepar2 network dataset
├── LEGAL/                    # Legal dataset variants
└── Lucas/                    # Lucas dataset
```

## Dataset Categories

### 1. Real-World Benchmark Datasets

#### DREAM4 Challenge Datasets
- **dream41/**, **dream42/**, **dream43/**, **dream44/**
- **Source**: DREAM4 In Silico Network Challenge
- **Description**: Synthetic gene regulatory networks with known topology
- **Files per dataset**:
  - `dream4X.csv`: Gene expression data
  - `adj.npy`: Ground truth adjacency matrix
  - `dag.gexf`: Graph representation
  - `nodes.npy`: Node information

#### Sachs Dataset
- **Location**: `sachs/`
- **Description**: Protein signaling network data
- **Features**: 11 proteins, 7466 observations
- **Files**:
  - `sachs.csv`: Protein expression data
  - `adj.npy`: Ground truth adjacency matrix
  - `dag.gexf`: Graph representation
  - `nodes.npy`: Node information

#### Bayesian Network Datasets
- **asia/**: Asia network (8 nodes, 8 edges)
- **alarm/**: ALARM network (37 nodes, 46 edges)
- **Hepar2/**: Hepar2 network (70 nodes, 123 edges)

### 2. Synthetic Datasets

#### BIAS Dataset
- **Location**: `BIAS/`
- **Description**: Synthetic datasets with different noise characteristics
- **Variants**:
  - `gaussian_i2e.csv`: Gaussian noise, i2e configuration
  - `gaussian_n2e.csv`: Gaussian noise, n2e configuration
  - `gaussian_n2i.csv`: Gaussian noise, n2i configuration
  - `non-gaussian_*`: Non-Gaussian noise variants
- **Adjacency Matrices**: `ADJ_MATRICES/` subdirectory

#### LEGAL Dataset
- **Location**: `LEGAL/`
- **Description**: Legal domain synthetic data
- **Variants**:
  - `gaussian_legal.csv`: Gaussian noise variant
  - `non-gaussian_legal.csv`: Non-Gaussian noise variant
- **Ground Truth**: `ADJ_MATRICES/adj_matrix_legal.csv`

### 3. Specialized Datasets

#### Lucas Dataset
- **Location**: `Lucas/`
- **Description**: Lucas dataset with specific domain characteristics
- **Files**:
  - `lucas.csv`: Dataset observations
  - `adj_matrix_lucas.npy`: Ground truth adjacency matrix

## File Formats

### Data Files (.csv)
- **Format**: Comma-separated values
- **Structure**: Rows = samples, Columns = variables
- **Headers**: Variable names or indices
- **Missing Values**: Handled according to dataset specifications

### Adjacency Matrices (.npy)
- **Format**: NumPy binary format
- **Structure**: Square matrix where `adj[i,j] = 1` indicates edge from node i to node j
- **Type**: Binary (0/1) or weighted (continuous values)
- **Size**: N×N where N is the number of variables

### Graph Files (.gexf)
- **Format**: GEXF (Graph Exchange XML Format)
- **Usage**: Network visualization and analysis
- **Tools**: Compatible with Gephi, Cytoscape, NetworkX

### Node Information (.npy)
- **Format**: NumPy array
- **Content**: Node labels, names, or metadata
- **Usage**: Human-readable variable identification


### Loading a Dataset

```python
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('Data/asia/asia.csv')
adj_matrix = np.load('Data/asia/adj.npy')
nodes = np.load('Data/asia/nodes.npy')

print(f"Data shape: {data.shape}")
print(f"Adjacency matrix shape: {adj_matrix.shape}")
print(f"Number of edges: {np.sum(adj_matrix)}")
```

### Working with Multiple Datasets

```python
import os

# List all available datasets
data_dir = 'Data'
datasets = [d for d in os.listdir(data_dir) 
           if os.path.isdir(os.path.join(data_dir, d))]

for dataset in datasets:
    data_path = f'Data/{dataset}/{dataset}.csv'
    if os.path.exists(data_path):
        print(f"Found dataset: {dataset}")
```

### Converting Formats

```python
# Convert adjacency matrix to CSV
adj_matrix = np.load('Data/asia/adj.npy')
pd.DataFrame(adj_matrix).to_csv('adjacency_matrix.csv', 
                               index=False, header=False)
```

## Citation

When using these datasets, please cite the original sources:

- **DREAM4**: [Marbach et al., 2009]
- **Sachs**: [Sachs et al., 2005]
- **Asia/Alarm**: [Lauritzen & Spiegelhalter, 1988]
- **Hepar2**: [Onisko et al., 2001]

## Notes

- All datasets are preprocessed and ready for causal discovery algorithms
- Ground truth adjacency matrices are binary (0/1) unless otherwise specified


