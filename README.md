# Causal-LLM-Framework-for-Graph-Discovery

A comprehensive framework for evaluating Large Language Models (LLMs) in causal discovery tasks, combining domain knowledge utilization with data-driven approaches to uncover causal relationships in complex systems.

## Overview

This framework explores whether Large Language Models can uncover causal relationships by leveraging embedded domain knowledge and identifying statistical patterns. The project integrates two key strategies:

1. **Domain Knowledge Approach**: Using LLMs' embedded knowledge through prompt-based full-graph discovery
2. **Data-Driven Approach**: Employing multivariate statistical dependency modeling (causal_llm)

The framework rigorously evaluates the effectiveness of LLMs in causal discovery, comparing them against traditional data-driven algorithms across various scenarios.

![diag2 drawio](https://github.com/user-attachments/assets/
4ac1172b-780d-49af-a044-430c03a7d359)

## Project Structure

```
causal_llm/
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── Dag_generation and model_evaluation/  # Model training and evaluation
├── Data/                              # Benchmark datasets
└── Dataset_generation/                # Synthetic data generation tools
```

## Key Features

- **Multi-Model Evaluation**: Compare LLM-based causal discovery against traditional algorithms (PC, GES, ICALiNGAM, GraNDAG, RL)
- **Comprehensive Datasets**: Support for both synthetic and real-world benchmark datasets
- **Flexible Architecture**: Modular design supporting different LLM backends (LLaMA, GPT, Gemma, DeepSeek)
- **Statistical Validation**: Robust evaluation metrics for causal graph discovery
- **Synthetic Data Generation**: Tools for generating controlled datasets with known ground truth

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data**:
   ```bash
   cd Dataset_generation
   python run.ipynb
   ```

3. **Run Model Evaluation**:
   ```bash
   cd "Dag_generation and model_evaluation"
   python main_runner.py
   ```

## Datasets

The framework includes several benchmark datasets:
- **Synthetic Datasets**: Linear, non-linear, and Gaussian Process-based data
- **Real-world Datasets**: DREAM4, Sachs, Asia, Andes, Alarm, and more
- **Custom Datasets**: Support for user-defined datasets

## Models Supported

- **Causal LLM**: Custom LLM-based causal discovery model
- **Traditional Methods**: PC, GES, ICALiNGAM, GraNDAG, RL
- **Hybrid Approaches**: Combining LLM knowledge with statistical methods

## Evaluation Metrics

- Structural Hamming Distance (SHD)
- Precision and Recall
- F1 Score
- Edge accuracy
- Graph similarity measures


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.
