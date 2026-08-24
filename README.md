# 🎬 Capstone Project — Movie Review Sentiment Analysis (MLOps)

A production-grade, end-to-end **MLOps pipeline** for binary sentiment classification (Positive / Negative) on movie reviews. The project goes from raw data to a live, monitored, auto-scaling web service — training a sentiment model along the way and covering data versioning, experiment tracking, CI/CD, containerization, and Kubernetes deployment on AWS.

---

## 📖 Overview

This project trains a TF-IDF + Logistic Regression model to classify movie reviews as **positive** or **negative**, then wraps it in a Flask web app for real-time inference. What makes it a "Capstone" MLOps project is everything *around* the model:

- Versioned, reproducible data pipelines with **DVC**
- Experiment tracking and a model registry with **MLflow** (hosted on DagsHub)
- Automatic model promotion from `Staging` → `Production` after passing tests
- A fully automated **CI/CD pipeline** (GitHub Actions) that tests, builds, and ships the app
- **Docker**-based containerization served with Gunicorn
- Deployment to **AWS EKS** (Kubernetes) behind a LoadBalancer, with images stored in **AWS ECR**
- Built-in **Prometheus** metrics for observability, ready to be visualized in **Grafana**

## 🏗️ Architecture

```text
      ┌────────────────────────┐
      │   Code Push (GitHub)   │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │  GitHub Actions CI/CD  │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Data Ingestion (S3)  │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Data Preprocessing   │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Feature Engineering  │
      │        (TF-IDF)        │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │      Model Training    │
      │  (Logistic Regression) │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │    Model Evaluation    │
      │   (MLflow / DagsHub)   │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Model Registration   │
      │        (Staging)       │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │    Model Promotion     │
      │      (Production)      │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Docker Image Build   │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │    Push to AWS ECR     │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Deploy to AWS EKS    │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │       Flask App        │
      │  (Serves Predictions)  │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │   Prometheus Metrics   │
      └────────────┬───────────┘
                    ↓
      ┌────────────────────────┐
      │    Grafana Dashboard   │
      └─────────────────────────┘
```

The whole flow (ingestion → preprocessing → features → training → evaluation → registration) is orchestrated as a **DVC pipeline** (`dvc.yaml`), and the whole flow from code push to a live Kubernetes rollout is orchestrated by a **single GitHub Actions workflow** (`.github/workflows/ci.yaml`).

## ✨ Key Features

- **Reproducible ML pipeline** — every stage (ingestion, preprocessing, feature engineering, training, evaluation, registration) is defined as a DVC stage with explicit dependencies, parameters, and outputs.
- **Experiment tracking & model registry** — MLflow (backed by DagsHub) logs metrics, parameters, and model artifacts for every run, and manages model lifecycle stages (`Staging` → `Production` → `Archived`).
- **Automated model promotion** — `scripts/promote_model.py` promotes the latest staged model to production only after it passes the model test suite.
- **Text preprocessing** — lowercasing, URL/number/punctuation stripping, stopword removal, and lemmatization (NLTK) shared between the training pipeline and the live inference path.
- **Web UI + REST-style prediction endpoint** — a lightweight Flask app (`flask_app/app.py`) serves a simple form-based UI and a `/predict` route.
- **Observability out of the box** — custom Prometheus counters/histograms track request volume, latency, and prediction class distribution via a `/metrics` endpoint.
- **Containerized & production-served** — Docker image runs the app with Gunicorn, not the Flask dev server.
- **CI/CD automation** — on every push, GitHub Actions reproduces the DVC pipeline, runs model + app tests, promotes the model, builds/pushes the Docker image to ECR, and rolls out the new version to an EKS cluster.
- **Cloud-native storage** — raw dataset lives in S3; trained artifacts (model, vectorizer) are versioned and shipped inside the Docker image.

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data pipeline | DVC, PySpark-free pandas/NumPy processing |
| ML | scikit-learn (TF-IDF, Logistic Regression), NLTK |
| Experiment tracking / registry | MLflow, DagsHub |
| Web framework | Flask, Gunicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS S3 (data), AWS ECR (image registry), AWS EKS (Kubernetes cluster) |
| Monitoring | Prometheus (metrics), Grafana (dashboards) |
| Testing | unittest |

## 📁 Project Structure

```
Capstone-Project/
├── .github/workflows/ci.yaml     # CI/CD pipeline: test → train → promote → build → deploy
├── src/
│   ├── connections/
│   │   └── s3_connection.py      # Fetches raw dataset from AWS S3
│   ├── data/
│   │   ├── data_ingestion.py     # Downloads data, splits train/test, saves to data/raw
│   │   └── data_preprocessing.py # Cleans & lemmatizes review text -> data/interim
│   ├── features/
│   │   └── feature_engineering.py# TF-IDF vectorization -> data/processed, models/vectorizer.pkl
│   ├── model/
│   │   ├── model_building.py     # Trains Logistic Regression -> models/model.pkl
│   │   ├── model_evaluation.py   # Computes metrics, logs to MLflow -> reports/
│   │   └── register_model.py     # Registers model in MLflow Model Registry (Staging)
│   └── logger/                   # Shared rotating-file + console logger
├── scripts/
│   └── promote_model.py          # Promotes Staging model to Production in MLflow
├── flask_app/
│   ├── app.py                    # Flask web app + Prometheus metrics + inference
│   ├── requirements.txt
│   └── templates/index.html      # Simple prediction UI
├── tests/
│   ├── test_model.py             # Validates loaded production model & performance thresholds
│   └── test_flask_app.py         # Validates Flask routes
├── notebooks/                    # Exploratory experiments (BoW vs TF-IDF, hyperparameter search)
├── dvc.yaml / dvc.lock           # DVC pipeline stage definitions
├── params.yaml                   # Pipeline hyperparameters (test_size, max_features)
├── Dockerfile                    # Builds the production Flask app image
├── deployment.yaml               # Kubernetes Deployment + LoadBalancer Service for EKS
├── requirements.txt              # Full pipeline/dev dependencies
├── setup.py                      # Makes `src` an installable local package
└── Makefile                      # Convenience commands (setup, lint, sync to S3, etc.)
```

## 🔄 DVC Pipeline Stages

| Stage | Command | Depends on | Produces |
|---|---|---|---|
| `data_ingestion` | `python src/data/data_ingestion.py` | S3 dataset | `data/raw/` |
| `data_preprocessing` | `python src/data/data_preprocessing.py` | `data/raw/` | `data/interim/` |
| `feature_engineering` | `python src/features/feature_engineering.py` | `data/interim/` | `data/processed/`, `models/vectorizer.pkl` |
| `model_building` | `python src/model/model_building.py` | `data/processed/` | `models/model.pkl` |
| `model_evaluation` | `python src/model/model_evaluation.py` | `models/model.pkl` | `reports/metrics.json`, `reports/experiment_info.json` |
| `model_registration` | `python src/model/register_model.py` | `reports/experiment_info.json` | Model registered in MLflow (Staging) |

Run the whole pipeline with a single command (see [Usage](#-usage) below).

## ✅ Prerequisites

- Python 3.10
- Docker (for containerized runs/deployment)
- An AWS account with access to S3, ECR, and EKS, plus configured credentials
- A [DagsHub](https://dagshub.com/) account for MLflow tracking (used as the `CAPSTONE_TEST` token)
- `kubectl` and `aws-cli` (only needed for manual Kubernetes deployment)

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArupaBarua/Capstone-Project.git
   cd Capstone-Project
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set required environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID=your_aws_access_key
   export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   export CAPSTONE_TEST=your_dagshub_access_token   # used for MLflow tracking auth
   ```

4. **(Optional) Pull DVC-tracked data/artifacts** if a DVC remote is configured
   ```bash
   dvc pull
   ```

## 🚀 Usage

### Run the full ML pipeline
Reproduces every DVC stage from data ingestion through model registration:
```bash
dvc repro
```

### Promote the trained model to Production
```bash
python scripts/promote_model.py
```

### Run the Flask app locally
```bash
cd flask_app
python app.py
```
The app will be available at `http://localhost:5000`.

### Run with Docker
```bash
docker build -t capstone-sentiment-app .
docker run -p 5000:5000 -e CAPSTONE_TEST=your_dagshub_token capstone-sentiment-app
```
The Docker image copies the Flask app and the trained vectorizer, installs dependencies, downloads the required NLTK corpora, and serves the app with **Gunicorn** in production mode.

## 🌐 API / Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the sentiment prediction form |
| `/predict` | POST | Accepts a `text` form field, returns the predicted sentiment (Positive/Negative) |
| `/metrics` | GET | Exposes Prometheus-formatted metrics (request count, latency, prediction distribution) |

## 🧪 Testing

```bash
# Validate the production model (load, signature, performance thresholds)
python -m unittest tests/test_model.py

# Validate Flask app routes
python -m unittest tests/test_flask_app.py
```

## 🔁 CI/CD Pipeline

Every push triggers the GitHub Actions workflow in `.github/workflows/ci.yaml`, which:

1. Checks out the code and sets up Python 3.10 (with pip caching)
2. Installs dependencies and runs `dvc repro` to reproduce the full pipeline
3. Runs the model test suite (`tests/test_model.py`)
4. Promotes the newly trained model from Staging to Production
5. Runs the Flask app test suite (`tests/test_flask_app.py`)
6. Authenticates to **AWS ECR** and builds the Docker image
7. Tags and pushes the image to ECR
8. Configures `kubectl` for the target **EKS** cluster
9. Creates/updates a Kubernetes secret with the DagsHub token
10. Applies `deployment.yaml` (with environment substitution) to roll out the new image

## ☸️ Kubernetes Deployment

`deployment.yaml` defines:
- A **Deployment** running 2 replicas of the Flask app container, pulling the image from ECR, with CPU/memory requests & limits, and the DagsHub token injected from a Kubernetes Secret.
- A **LoadBalancer Service** exposing the app on port `5000`.

Manual deploy (if not going through CI/CD):
```bash
aws eks update-kubeconfig --region us-east-1 --name flask-app-cluster
kubectl create secret generic capstone-secret --from-literal=CAPSTONE_TEST=$CAPSTONE_TEST
envsubst < deployment.yaml | kubectl apply -f -
```

## 📊 Monitoring

The Flask app exposes a `/metrics` endpoint instrumented with `prometheus_client`, tracking:
- Total requests per method/endpoint
- Request latency per endpoint
- Prediction counts per sentiment class

Point a **Prometheus** server at this endpoint to scrape metrics, then build **Grafana** dashboards on top for real-time visibility into traffic and model behavior in production.

## 📓 Notebooks

The `notebooks/` directory contains exploratory work used to arrive at the production pipeline's design choices, including Bag-of-Words vs. TF-IDF comparisons and Logistic Regression hyperparameter tuning.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Arupa Barua**

![CI Pipeline](https://github.com/ArupaBarua/Capstone-Project/actions/workflows/ci.yaml/badge.svg)
