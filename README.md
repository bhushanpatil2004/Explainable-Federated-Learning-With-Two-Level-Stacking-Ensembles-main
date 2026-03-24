# Explainable Federated Learning With Two-Level Stacking Ensembles

Advanced federated learning framework for heart disease prediction using explainable AI and ensemble methods.

## Overview

This project implements a sophisticated federated learning system that combines multiple machine learning models through two-level stacking ensembles while maintaining model explainability through SHAP values.

## Features

- **Federated Learning**: Privacy-preserving distributed model training
- **Two-Level Stacking**: Advanced ensemble technique combining multiple base models
- **Explainable AI**: SHAP-based feature importance and model interpretability
- **Multiple Clients**: Support for training across multiple data sources
- **Advanced Preprocessing**: Comprehensive data cleaning and feature engineering

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Train federated model
python advanced_train_model.py --client_id client_1 --target_accuracy 0.95
```

## Project Structure

```
project/
│   └── advanced_train_model.py    # Main federated learning model
|--- advanced_preprocesses.py
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore patterns
└── README.md                      # Project documentation
```

## License

MIT License
