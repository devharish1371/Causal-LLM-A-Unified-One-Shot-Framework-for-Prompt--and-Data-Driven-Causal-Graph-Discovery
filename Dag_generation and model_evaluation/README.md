# DAG Generation and Model Evaluation

This module contains the core implementation for training and evaluating causal discovery models, including the novel Causal LLM approach and traditional baseline methods.

## Overview

The module provides a comprehensive framework for:
- Training LLM-based causal discovery models
- Evaluating multiple causal discovery algorithms
- Comparing model performance against ground truth DAGs
- Generating and saving adjacency matrices for discovered causal graphs

## Files Description

### Core Implementation

- **`causal_llm.py`**: Contains the main CausalDiscoveryLLM class implementing the LLM-based causal discovery approach
- **`models.py`**: Model initialization and training utilities for all supported algorithms
- **`config_models.py`**: Configuration parameters for different causal discovery models
- **`main_runner.py`**: Main execution script for running model evaluation pipeline
- **`evaluation.py`**: Evaluation metrics and result analysis functions
- **`data_loader.py`**: Data loading utilities for different dataset formats
- **`convert_npy_to_csv.py`**: Utility for converting NumPy arrays to CSV format

### Demo and Documentation

- **`run.ipynb`**: Jupyter notebook demonstrating model usage and evaluation
- **`requirements.txt`**: Python dependencies specific to this module

## Key Components

### CausalDiscoveryLLM Class

The main class implementing the LLM-based causal discovery approach:

```python
class CausalDiscoveryLLM:
    def __init__(self, input_dim, output_dim, model_path=None)
    def learn(self, data, num_epochs=10, batch_size=32, epsilon=0.1)
    def causal_matrix(self, data)
```

**Features:**
- Uses LLaMA/GPT-based architecture with frozen pre-trained weights
- Implements epsilon-greedy exploration strategy
- Includes L1 regularization for sparsity
- Ensures acyclic graph generation
- Supports graph pruning based on linear regression coefficients

### Supported Models

1. **Causal LLM**: Custom LLM-based approach
2. **PC**: PC algorithm for causal discovery
3. **GES**: Greedy Equivalence Search
4. **ICALiNGAM**: Independent Component Analysis-based LiNGAM
5. **GraNDAG**: Gradient-based Neural DAG learning
6. **RL**: Reinforcement Learning-based approach

### Model Configuration

The `config_models.py` file contains hyperparameters for each model:

```python
config = {
    "causal_llm": {
        "input_dim": None,  
        "output_dim": lambda input_dim: input_dim * input_dim,
        "model_path": "llm_model.pth"
    },
    "RL": {"nb_epoch": 100},
    "ICALiNGAM": {"max_iter": 10000, "thresh": 0.1},
    "GraNDAG": {"input_dim": None, "iterations": 1000},
    "GES": {},
    "PC": {}
}
```

## Usage

### Basic Usage

```python
from main_runner import main

# Run evaluation on a dataset
main(
    ground_truth_path="path/to/ground_truth.csv",
    gaussian_path="path/to/gaussian_data.csv",
    non_gaussian_path="path/to/non_gaussian_data.csv"
)
```

### Custom Model Training

```python
from causal_llm import CausalDiscoveryLLM
import numpy as np

# Initialize model
model = CausalDiscoveryLLM(
    input_dim=10,
    output_dim=100,
    model_path="model.pth"
)

# Load your data
data = np.random.randn(1000, 10)

# Train the model
model.learn(data, num_epochs=50, batch_size=64)

# Get causal adjacency matrix
adj_matrix = model.causal_matrix(data)
```

### Running Individual Models

```python
from models import initialize_models, train_models

# Initialize all configured models
models = initialize_models(input_dim=10)

# Train models on data
results = train_models(models, data, dataset_type="gaussian", 
                      node_labels=None, ground_truth_name="test")
```

## Key Features

### 1. Acyclic Graph Generation
- Implements `ensure_acyclic()` function to guarantee DAG structure
- Removes lowest-weight edges in cycles iteratively
- Uses NetworkX for cycle detection

### 2. Graph Pruning
- `graph_prunned_by_coef()` function prunes edges based on linear regression coefficients
- Threshold-based edge removal for sparsity
- Maintains causal structure while reducing noise

### 3. Flexible Architecture
- Supports multiple LLM backends (LLaMA, GPT, Gemma, DeepSeek)
- Configurable model parameters
- Easy addition of new causal discovery algorithms

### 4. Comprehensive Evaluation
- Multiple evaluation metrics
- Comparison against ground truth
- Statistical significance testing
- Result visualization and export

## Dependencies

- `torch`: PyTorch for neural network implementation
- `transformers`: Hugging Face transformers for LLM models
- `networkx`: Graph manipulation and cycle detection
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: Linear regression for graph pruning
- `gcastle`: Causal discovery algorithms
- `tqdm`: Progress bars

## Configuration

### Model Parameters

- **Learning Rate**: 2e-5 (Adam optimizer)
- **Batch Size**: 32 (configurable)
- **Epochs**: 10 (configurable)
- **Epsilon**: 0.1 (exploration rate)
- **L1 Regularization**: 0.01

### Architecture Parameters

- **Hidden Size**: 512
- **Intermediate Size**: 1024
- **Num Hidden Layers**: 8
- **Num Attention Heads**: 8
- **Max Position Embeddings**: 512

## Output

The module generates:
- Adjacency matrices for each model in CSV format
- Evaluation metrics and comparison results
- Trained model checkpoints (for Causal LLM)
- Performance statistics and visualizations

## Example Output Files

- `{dataset_name}_{model_name}_{type}.csv`: Adjacency matrices
- `evaluation_results.json`: Comprehensive evaluation metrics
- `model_checkpoints/`: Trained model files

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Convergence**: Increase epochs or adjust learning rate
3. **Graph Cycles**: Check `ensure_acyclic()` implementation
4. **Data Format**: Ensure data is properly normalized and formatted

### Performance Tips

- Use GPU acceleration when available
- Adjust batch size based on available memory
- Monitor training loss for convergence
- Use appropriate data preprocessing

## Contributing

When adding new models:
1. Add configuration to `config_models.py`
2. Implement model class in `models.py`
3. Update `initialize_models()` function
4. Add evaluation metrics if needed
5. Update documentation

