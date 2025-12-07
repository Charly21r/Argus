# Multimodal Content Moderation System
A scalable, multimodal content-moderation system engineered for real-time, production-grade toxicity and hate-speech detection.

## Overview
The system analyzes:
* Text-only content (e.g., comments, posts, messages)
* Image-only content (e.g., screenshots, photos) _(in progress)_
* Multimodal content (e.g., memes combining text + images) _(in progress)_

All models adhere to a unified output schema, enabling seamless downstream integration in safety pipelines, Trust & Safety dashboards, real-time moderation services, and enterprise workflows.

The repo includes:
- End-to-end training & evaluation pipelines
- Modular dataset ingestion (text, image, multimodal)
- Bias & fairness evaluation with CI enforcement
- Threshold calibration and slice-based metrics
- Artifact management with MLflow
- Templated synthetic datasets for robust lexical-bias auditing


## Model Architecture
Three independent models are trained and evaluated:

### 1. Text-Only Moderation Model
Uses transformer-based architectures (e.g, DistilBERT) to predict:
- toxicity
- hate

Training includes:
- Multi-label classification
- Class imbalance handling (pos_weight)
- Threshold calibration for F1 optimization
- Lexical bias slice evaluation

### 2. Image-Only Moderation Model
CNN / Vision Transformer–based classifier predicting image-based hate signals.

(Implementation in progress.)

### 3. Multimodal Moderation Model (Memes)
Joint text + image encoder for memes and mixed-modality content.

(Implementation in progress.)

Supports datasets such as:
- Facebook Hateful Memes
- MMHS150K



## Unified Ouput Schemas
Text-only schema:
```
{
    "id": str,
    "text": str,
    "toxicity": int,  # 0/1
    "hate": int,      # 0/1
    "source": str,    # 'jigsaw'
}
```

Multimodal schema:
```
{
    "id": str,
    "text": str,
    "image_path": str,  # relative path under data/raw/...
    "hate": int,        # 0/1
    "source": str,      # 'hateful_memes' or 'mmhs150k'
}
```


## Datasets
The project supports (and normalizes) the following datasets:
1. Jigsaw Toxic Comments (text-only)
2. MMHS150K (multimodal)
3. Facebook Hateful Memes Dataset (multimodal)

The system is designed so additional datasets can be added with minimal boilerplate.


## Bias, Fairness & Responsible AI Evaluation
Content-moderation models are prone to lexical bias, e.g.:
- Texts mentioning certain groups such as “muslim”, “christian”, “gay”, “woman”, etc. being disproportionately predicted as toxic.

This repo includes a fairness evaluation pipeline, consisting of:

### 1. Synthetic Templated Bias Dataset
Generated via:
```
scripts/generate_bias_templates.py
```
Creates controlled toxic and non-toxic examples. This enables precise slice-level FPR/TPR measurement.

### 2. Automated Bias Evaluation Job
```
scripts/run_bias_eval.py
```

Runs the trained model against:
- Real validation data slices
- Synthetic templated examples

Outputs:
```
models/text_toxicity/artifacts/bias_report.json
```

Which contains, for each group:
- FPR / TPR
- ROC-AUC / PR-AUC
- Slice sample counts
- $\Delta$-metrics (e.g. $\Delta_{FPR}$ between group & non-group)


## 3. CI Fairness Gate
```
tests/test_bias_constraints.py
```

This enforces fairness constraints such as:
- No group’s FPR delta may exceed 5 percentage points.
- No extreme divergence in TPR between groups.

A failing fairness test blocks the model from being promoted


## Training Pipeline
Key components:
- Structured PyTorch Lightning–style modular training loop
- MLflow tracking for reproducibility
- Seeded runs for determinism
- Custom:
    - eval_model()
    - compute_metrics()
    - find_optimal_thresholds()
- Centralized metric computation for both:
    - overall performance
    - fairness slice performance



## Repo Structure
```
.
├── config/
│   └── local_sensitive_words.json
├── data/
│   ├── preprocessed/
│   │   └── text/
│   ├── raw/
│   │   └── jigsaw/
│   └── bias_eval/
├── models/
│   └── text_toxicity/
│       └── artifacts/
├── scripts/
│   ├── generate_bias_templates.py
│   └── run_bias_eval.py
├── src/
│   ├── data/
│   │   └── jigsaw_preprocessing.py
│   ├── monitoring/
│   ├── serving/
│   └── training/
│       └── train_text_model.py
├── notebooks/
│   ├── eda_jigsaw.py
│   └── experiment.py
└── tests/
    └── test_bias_constraints.py
```
