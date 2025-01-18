# Domino Drift Effect Approach for Probability Estimation of Feature Drift in High-dimensional Data

## Overview
This project implements a novel approach called "Domino Drift Effect" for detecting and estimating feature drift probabilities in high-dimensional datasets. The implementation includes experiments on multiple datasets including:
- CovType
- CICIDS
- Insects
- Heartbeats

## Requirements
The project requires Python 3.10+

## Project Structure

### Notebooks
The project contains several Jupyter notebooks for different experiments:
- `notebooks/drift_score_correlation_covertype.ipynb`: Experiments on the CovType dataset
- `notebooks/drift_score_correlation_cicids.ipynb`: Experiments on the CICIDS dataset
- `notebooks/drift_score_correlation_insect.ipynb`: Experiments on the Insects dataset
- `notebooks/drift_score_correlation_heartbeats.ipynb`: Experiments on the Heartbeats dataset
- `notebooks/monitored_feat_eval.ipynb`: Evaluation of monitored features
- `notebooks/nonmonitored_feat_eval.ipynb`: Evaluation of non-monitored features

### Core Components
- `test_harness/datasets/`: Contains dataset handling code
- `test_harness/experiments/`: Contains experiment implementation code including the DDE (Domino Drift Effect) experiment

## Setup and Installation

1. Clone the repository
2. Install the required dependencies using `pip install -r requirements.txt`
3. Run the notebooks using Jupyter Notebook or Jupyter Lab

## Running Experiments

1. Ensure your data files are placed in the appropriate directories:
   - For CICIDS dataset: `data/cic_0.01km.csv`
   - Other datasets should be placed according to the notebook configurations

2. Open and run the relevant Jupyter notebooks.


3. Navigate to the `notebooks/` directory and select the appropriate notebook for your experiment.

## Key Features

- Implementation of the Domino Drift Effect approach for drift detection
- Support for multiple datasets and experiment configurations
- Comprehensive evaluation metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Cross-validation capabilities
- Visualization tools for drift analysis

## Notes

- The project uses various drift detection methods including KS (Kolmogorov-Smirnov) test and ADWIN
- Experiment results are typically saved in Excel format for further analysis
- The codebase includes functionality for both monitored and non-monitored feature evaluation
