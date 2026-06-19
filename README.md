<div align="center">

<img src="assets/argus_logo.svg" alt="Argus" width="520"/>

[![CI](https://github.com/Charly21r/content-moderation-system/actions/workflows/ci.yaml/badge.svg)](https://github.com/Charly21r/content-moderation-system/actions/workflows/ci.yaml)
[![Model Validation](https://github.com/Charly21r/content-moderation-system/actions/workflows/model-validation.yaml/badge.svg)](https://github.com/Charly21r/content-moderation-system/actions/workflows/model-validation.yaml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

Argus is a production-ready content moderation API that detects toxicity and hate speech in real time, with bias mitigation and fairness constraints enforced across the full ML pipeline.

---

## Overview

Argus detects toxic and hateful content across three modalities:

| Modality | Status |
|---|---|
| Text (comments, posts, messages) | **Available** |
| Image (screenshots, photos) | In progress |
| Multimodal (memes: text + image) | In progress |

---

## Key Features

- **Multi-label classification** — simultaneous toxicity + hate detection per input
- **Transformer-based text model** — fine-tuned on Jigsaw Toxic Comments
- **Precision-at-recall threshold calibration** — production-oriented thresholds, not hard 0.5 defaults
- **Fairness-aware training** — counterfactual data augmentation to reduce lexical bias
- **Fairness evaluation** — slice-level FPR/TPR across identity groups with CI enforcement
- **MLflow tracking** — full reproducibility with artifact and metric logging; Databricks-hosted registry supported
- **Dual inference backends** — PyTorch or ONNX Runtime (INT8 quantized)
- **REST API** — FastAPI with single and batch endpoints
- **Containerized** — Docker + Docker Compose (API + MLflow + Prometheus + Grafana)
- **Observability** — Prometheus metrics + Grafana dashboards

---

## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Charly21r/content-moderation-system
cd content-moderation-system
pip install -e .
```

Copy the environment template and fill it in:

```bash
cp .env.example .env
```

---

## Configuration

Argus uses two config files with a clear separation of concerns:

**`config/training.yaml`** — committed to the repo, shared defaults for model, training, paths, and serving. You rarely need to edit this.

**`.env`** — gitignored, your personal credentials and environment-specific overrides. Always required.

### Minimal `.env` (local only, no Databricks)

```env
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=argus
```

No credentials needed. The model is loaded from `models/text_toxicity/artifacts/` locally.

### Full `.env` (with Databricks or other Model Registry)

```env
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_NAME=/Users/you@email.com/argus
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=your-token

# Pull model from Databricks registry at startup instead of local folder
CMS_SERVING__MODEL_URI=models:/workspace.default.content-moderation-text@Production
```

**Any value in `training.yaml` can be overridden with a `CMS_`-prefixed env var, e.g. `CMS_TRAINING__EPOCHS=5`.**

---

## Serving

### PyTorch backend (default)

```bash
uvicorn src.serving.app:app
```

On first start with `CMS_SERVING__MODEL_URI` set, the model is downloaded from the Model Registry. Subsequent starts use the cached download.

Without `CMS_SERVING__MODEL_URI`, the model is loaded from `models/text_toxicity/artifacts/` locally.

### ONNX backend (faster inference)

First export the registered model to ONNX:

```bash
python -m src.optimization.export_onnx
```

This downloads the `Production` PyTorch model from the registry, converts it to ONNX FP32, validates numerical equivalence, and does two things:
- Writes the ONNX model locally to `models/text_toxicity/onnx/`
- Registers it in the MLflow Model Registry as `content-moderation-text-onnx`

Serve with the ONNX backend — local:

```bash
CMS_SERVING__BACKEND=onnx CMS_SERVING__MODEL_URI=models/text_toxicity/onnx uvicorn src.serving.app:app
```

Or directly from the registry:

```bash
CMS_SERVING__BACKEND=onnx CMS_SERVING__MODEL_URI="models:/workspace.default.content-moderation-text-onnx@Production" uvicorn src.serving.app:app
```

Re-run the export whenever you promote a new PyTorch model version. Then set the `Production` alias on the new ONNX version in the Databricks UI.

### Docker Compose (full stack)

```bash
docker compose up
```

| Service | URL |
|---|---|
| Argus API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/health` | Liveness check |
| `GET` | `/v1/model/info` | Loaded model metadata and thresholds |
| `POST` | `/v1/moderate/text` | Moderate a single text input |
| `POST` | `/v1/moderate/text/batch` | Moderate multiple texts in one request |
| `GET` | `/metrics` | Prometheus metrics |

### Single prediction

```bash
curl -X POST http://localhost:8000/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Your text here"}'
```

```json
{
  "id": "abc123",
  "text": "Your text here",
  "toxicity": { "prob": 0.03, "flagged": false },
  "hate":     { "prob": 0.01, "flagged": false },
  "safe": true,
  "processing_time_ms": 18.4
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/v1/moderate/text/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"content": "First text"}, {"content": "Second text"}]}'
```

---

## Training

### 1. Get the data

Download the [Jigsaw Toxic Comments dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and place `train.csv` at `data/raw/jigsaw/train.csv`.

### 2. Preprocess

```bash
python -m src.data.jigsaw_preprocessing
```

### 3. Train

```bash
python -m src.training.train_text_model
```

Metrics, parameters, and the model artifact are logged to MLflow automatically. Edit `config/training.yaml` to change hyperparameters.

### 4. Evaluate bias

```bash
python scripts/generate_bias_templates.py
python scripts/run_bias_eval.py
```

### 5. Calibrate thresholds (optional, no retraining needed)

```bash
python scripts/threshold_sweep.py --plot
```

Shows F1-optimal vs. precision-at-recall thresholds side by side for each label.

### 6. Promote to production

Open the Databricks MLflow UI, find your run, and set the alias `Production` on the model version you want to serve. The serving layer will pick it up on next startup.

---

## Optimization

### Export to ONNX

```bash
python -m src.optimization.export_onnx
```

Downloads the `Production` model from the registry, converts to ONNX FP32, validates numerical equivalence (max abs diff < 1e-4), and writes to `models/text_toxicity/onnx/`.

### Quantize to INT8

```bash
python -m src.optimization.quantize
```

Applies dynamic INT8 quantization and runs a bias comparison report (FP32 vs INT8) to verify that quantization doesn't disproportionately hurt fairness metrics.

---

## Fairness Pipeline

Content moderation models are prone to **lexical bias** — flagging content that mentions identity groups (e.g. "muslim", "gay") rather than detecting genuine toxicity.

Argus addresses this at every stage:

| Stage | Technique |
|---|---|
| Training | Counterfactual data augmentation (identity term swapping) |
| Evaluation | Slice-level FPR/TPR across identity groups |
| CI | Fairness gate blocks model promotion if ΔFPR > 5pp |
| Optimization | Post-quantization fairness comparison (FP32 vs INT8) |

---

## Development

```bash
make install     # install all deps in dev mode
make test        # run full test suite
make lint        # ruff check
make typecheck   # mypy
make format      # auto-format
make test-bias   # fairness constraint tests (requires model artifacts)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, HuggingFace Transformers |
| Inference | ONNX Runtime (optimum) |
| Training tracking | MLflow (local or Databricks) |
| Serving | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Observability | Prometheus, Grafana, OpenTelemetry |
| Testing | pytest |
| Linting / types | Ruff, mypy |
| CI | GitHub Actions |

---

## Repo Structure

```
.
├── config/
│   ├── training.yaml              # shared config defaults
│   └── local_sensitive_words.json
├── data/
│   ├── raw/jigsaw/
│   ├── preprocessed/text/
│   └── bias_eval/
├── models/
│   └── text_toxicity/
│       ├── artifacts/             # PyTorch model + thresholds (local)
│       ├── onnx/                  # ONNX FP32 export
│       └── quantized/             # ONNX INT8 export
├── monitoring/
│   ├── prometheus/
│   └── grafana/
├── reports/
├── scripts/
│   ├── generate_bias_templates.py
│   ├── run_bias_eval.py
│   └── threshold_sweep.py
├── src/
│   ├── data/
│   ├── optimization/
│   ├── serving/
│   ├── training/
│   └── utils/
├── tests/
├── .env.example                   # copy to .env and fill in
└── config/training.yaml
```
