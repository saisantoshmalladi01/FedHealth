# Federated Learning Project - CECS 574

## Overview
This project is part of the CECS 574 course on Topics in Distributed Computing. It demonstrates the implementation of a **federated learning** framework to predict diabetes using machine learning models.

## Project Structure
- **`server.py`**: Handles the server-side operations of the federated learning system.
- **`client.py`**: Simulates client-side computation and data sharing.
- **`neural.py`**: Contains neural network definitions and training logic.
- **`utils.py`**: Utility functions for data processing and model handling.
- **`diabetes_prediction_dataset.csv`**: Dataset used for training and evaluation.
- **`generate_confusion_matrix.py`**: Script for generating confusion matrices to evaluate predictions.
- **`logistic_XGBoast.py`**: Logistic regression and XGBoost model integration.

## Features
- Federated learning simulation with a focus on privacy-preserving data sharing.
- Implementation of multiple machine learning models for prediction tasks.
- Tools for model evaluation and visualization.

## How to Run
1. Ensure Python 3.x is installed along with necessary libraries (e.g., `numpy`, `pandas`, `scikit-learn`, `matplotlib`).
2. Run the server and client scripts to simulate federated learning:
