# Event Prediction in EMRs

This repository implements a two-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and Transformer-based sequence modeling to predict or impute patient events over time.

This model is a part of my thesis and will be used on actual EMR data, stored in a closed environment. For that, it is organized as a package that can be installed:

```bash
event-prediction-in-diabetes-care/
│
├── transform_emr/           # Core Python package
│   ├── config/              # Configuration modules
│   │   ├── dataset_config.py
│   │   └── model_config.py
│   │
│   ├── dataset.py           # Data preprocessing logic
│   ├── embedding.py         # Embedding model (EMREmbedding)
│   ├── transformer.py       # Transformer architecture
│   ├── train.py             # Training logic
│   ├── inference.py         # Inference pipeline
│   └── utils.py             # Utility functions
│
├── data/                      # External data folder (for synthetic or real EMR)
│   ├── train/
│   └── test/
│
├── tests/                     # Unit and integration tests
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🛠️ Installation

Install the project as an editable package from the **root** directory:

```bash
pip install -e .
```

```python
from transform_emr.dataset import EMRDataset
from transform_emr.train import run_two_phase_training
```

---

## 🚀 Usage

### 1. Prepare Dataset and Update Config

```python
import pandas as pd
from transform_emr.dataset import EMRDataset
from transform_emr.config import dataset_config, model_config

# Load data
df = pd.read_csv(dataset_config.TRAIN_TEMPORAL_DATA_FILE)
ctx_df = pd.read_csv(dataset_config.TRAIN_CTX_DATA_FILE)

# Initialize dataset and update config dynamically
ds = EMRDataset(df, ctx_df)
model_config.MODEL_CONFIG["vocab_size"] = len(ds.token2id)
model_config.MODEL_CONFIG["ctx_dim"] = ds.context_df.shape[1]
```

### 2. Train Model

```python
from transform_emr.train import run_two_phase_training
run_two_phase_training()
```

Model checkpoints and scaler are saved under `checkpoints/phase1/` and `checkpoints/phase2/`.

---

## 🧪 Running Tests

Run all tests:

```bash
pytest tests/
```

📝 If you have **not trained the model yet**, skip inference-related tests:

```bash
pytest tests/test_dataset.py tests/test_embedder.py
```

⚠️ Inference tests assume the presence of:
- `checkpoints/phase1/best_embedder.pt`
- `checkpoints/phase1/scaler.pkl`
- `checkpoints/phase2/best_transformer.pt`

To fix size mismatch errors:
```bash
rm -r checkpoints/
# Then rerun training
```

---

## 📦 Packaging Notes

To package without data/checkpoints:

```powershell
# Clean up any existing temp folder
Remove-Item -Recurse -Force .\transform_emr_temp -ErrorAction SilentlyContinue

# Recreate the temp folder
New-Item -ItemType Directory -Path .\transform_emr_temp | Out-Null

# Copy only what's needed
Copy-Item -Path .\transform_emr\* -Destination .\transform_emr_temp\transform_emr -Recurse
Copy-Item -Path .\setup.py, .\README.md, .\requirements.txt -Destination .\transform_emr_temp

# Zip it
Compress-Archive -Path .\transform_emr_temp\* -DestinationPath .\emr_model.zip -Force

# Clean up
Remove-Item -Recurse -Force .\transform_emr_temp
```

---

## 📌 Notes

- This project uses synthetic EMR data (`data/train/` and `data/test/`).
- For best results, ensure consistent preprocessing when saving/loading models.
- `model_config.py` should only be updated **after** dataset initialization to avoid embedding size mismatches.

---

## 🔄 End-to-End Workflow

Raw EMR Tables
│
▼
Per-patient Event Tokenization (with normalized timestamps)
│
▼
🧠 Phase 1 – Train EMREmbedding (token + time + patient context)
│
▼
📚 Phase 2 – Pre-train a Transformer decoder over learned embeddings, as a next-token-prediction task.
│
▼
→ Predict next medical events or missing timeline entries

---

## 📦 Module Overview

### 1. **`dataset.py`** – Temporal EMR Preprocessing

| Component            | Role                                                                                             |
|---------------------|--------------------------------------------------------------------------------------------------|
| `EMRDataset`        | Converts raw EMR tables into per-patient token sequences with relative time.                     |
| `_expand_tokens()`  | Generates CONCEPT_VALUE_(START/END) or single tokens from events. Tokenizing using (START/END) for time intervals allows to capture the length of an event (TIRP - a state or trend).                         |
| `collate_emr()`     | Pads sequences and returns tensors: `token_ids`, `time_deltas`, and fixed-length context vector. |

📌 **Why it matters:**  
Medical data varies in density and structure across patients. This dynamic preprocessing handles irregularity while preserving medically-relevant sequencing via `START/END` logic and relative timing.

---

### 2. **`embedding.py`** – EMR Representation Learning

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `Time2Vec`          | Learns periodic + trend encoding from inter-event durations.                                      |
| `EMREmbedding`      | Combines token, time, and patient context embeddings. Adds `[CTX]` token for global patient info. |
| `train_embedder()`  | Trains the embedding model with teacher-forced next-token prediction.                            |

🧠 **Insight:**  
Phase 1 learns a robust, patient-aware representation of their event sequences. It isolates the core structure of patient timelines without being confounded by the autoregressive depth of Transformers.

---

### 3. **`transformer.py`** – Causal Language Model over EMR Timelines

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `GPT`               | Transformer decoder stack over learned embeddings.                                                |
| `CausalSelfAttention` | Multi-head attention using causal mask to enforce chronology.                                 |
| `configure_optimizers()` | Groups model parameters for AdamW with correct weight decay policy.                         |

⚙️ **Phase 2: Learning Sequence Dependencies**  
Once the EMR structure is captured, the transformer learns to model sequential dependencies in event progression:  
- What tends to follow a certain event?  
- How does timing affect outcomes?  
- How does patient context modulate the trajectory?

---

## ✅ Model Capabilities

- ✔️ **Handles irregular time-series data** using relative deltas and Time2Vec.
- ✔️ **Captures both short- and long-range dependencies** with deep transformer blocks.
- ✔️ **Supports variable-length patient histories** using custom collate and attention masks.
- ✔️ **Imputes and predicts** events in structured EMR timelines.

---

## 📚 Citation
Inspired by recent advancements in temporal deep learning, sequence modeling in healthcare (BEHRT, RETAIN, Med-BERT), and Time2Vec (Kazemi et al.).