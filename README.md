# ❤️ Heart Disease Prediction using MLOps

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)]()
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)]()
[![Podman](https://img.shields.io/badge/Podman-Container-blue.svg)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-326CE5.svg)]()

---

# Project Overview

This project implements an **end-to-end MLOps pipeline** for predicting heart disease using the **UCI Heart Disease Dataset**.

The project demonstrates the complete machine learning lifecycle, including:

- Data acquisition
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Experiment tracking with MLflow
- Model packaging
- REST API development using FastAPI
- Podman containerization
- Kubernetes deployment
- Monitoring using Prometheus and Grafana
- CI/CD using GitHub Actions

---

# Dataset

Dataset: **Heart Disease Dataset**

Source:

https://raw.githubusercontent.com/plotly/datasets/master/heart.csv

Features include:

- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Maximum Heart Rate
- ST Depression
- Exercise-induced Angina
- Thalassemia
- Heart Disease Target

---

# Repository Structure

```text
heart-disease-mlops/

├── api/
├── artifacts/
├── data/
├── podman/
├── docs/
├── k8s/
├── monitoring/
├── notebooks/
├── screenshots/
├── src/
├── tests/
├── .github/
├── README.md
└── requirements.txt
```

---

# Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- MLflow
- FastAPI
- Docker
- Kubernetes
- Prometheus
- Grafana
- GitHub Actions
- Pytest

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/heart-disease-mlops.git

cd heart-disease-mlops
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Project Workflow

```
Download Dataset

↓

Data Preprocessing

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Model Training

↓

Hyperparameter Tuning

↓

MLflow Experiment Tracking

↓

Model Packaging

↓

FastAPI

↓

Docker

↓

Kubernetes

↓

Monitoring
```

---

# Running the Project

## Step 1

Download Dataset

```bash
python data/download_data.py
```

---

## Step 2

Run EDA Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Step 3

Feature Engineering

```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

---

## Step 4

Train Models

```bash
jupyter notebook notebooks/03_model_experiments.ipynb
```

---

## Step 5

Launch MLflow

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## Step 6

Run FastAPI

```bash
uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# Machine Learning Models

The following models were implemented:

- Logistic Regression
- Random Forest Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross Validation

The final model was selected based on the highest ROC-AUC score.

---

# MLflow Experiment Tracking

MLflow is used to track:

- Parameters
- Evaluation Metrics
- Trained Models
- Confusion Matrix
- ROC Curve
- Experiment History

---

# Podman

## Build Image

```bash
podman build -t heart-disease-api -f podman/Dockerfile .
```

Run Container

```bash
podman run -p 8000:8000 heart-disease-api
```

## GitHub Container Registry (GHCR)

Login to GHCR:

```bash
podman login ghcr.io -u YOUR_GITHUB_USERNAME
```

Build and tag for GHCR:

```bash
podman build -t ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-api:latest -f podman/Dockerfile .
```

Push to GHCR:

```bash
podman push ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-api:latest
```

Pull from GHCR:

```bash
podman pull ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-api:latest
```

---

# Kubernetes Deployment

Apply Kubernetes manifests

```bash
kubectl apply -f k8s/
```

Verify

```bash
kubectl get pods

kubectl get services

kubectl get ingress
```

---

# Monitoring

Monitoring tools used:

- Prometheus
- Grafana

Metrics collected:

- API Requests
- Response Time
- HTTP Status Codes
- Request Count

---

# Automated Testing

Run all tests

```bash
pytest
```

---

# CI/CD Pipeline

GitHub Actions automates:

- Code Checkout
- Dependency Installation
- Linting
- Unit Testing
- Model Training

Workflow files:

```
.github/workflows/ci.yml

.github/workflows/cd.yml
```

---

# API Endpoint

### POST `/predict`

Example Request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Example Response

```json
{
    "prediction": 1,
    "confidence": 0.96
}
```

---

# Screenshots

The following screenshots are included in the project:

```
screenshots/

├── eda/
├── mlflow/
├── github_actions/
├── podman/
├── kubernetes/
└── monitoring/
```

---

# Documentation

Project documentation is available in:

```
docs/

setup_guide.md

final_report.pdf

architecture_diagram.png
```

---

# Future Improvements

Potential future enhancements include:

- XGBoost and LightGBM models
- Automated model retraining
- Model drift detection
- Cloud deployment (AWS/GCP/Azure)
- Helm chart deployment
- Explainable AI using SHAP

---

# Repository

GitHub Repository:

```
https://github.com/<your-username>/heart-disease-mlops
```

Replace `<your-username>` with your GitHub username.

---

# Author

**Lalit Sharma**

MLOps Assignment – Heart Disease Prediction System

2026