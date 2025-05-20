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

# Ensure your working directory is properly set to the root repo of this project
# Be sure to set the path in your local env properly.
```

---

## 🚀 Usage

### 1. Prepare Dataset and Update Config

```python
import pandas as pd
from transform_emr.dataset import EMRDataset
from transform_emr.config.dataset_config import *
from transform_emr.config.model_config import *

# Load data (verify you paths are properly defined)
temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE, low_memory=False)

# You'll need to verify your temporal_df
# - Already has all the columns ['PatientID', 'ConceptName', 'Value', 'StartDateTime', 'EndDateTime']
# - Dates are in date-time (datetime64[ns]) format

# Initialize dataset and update config dynamically (based on the training set)
ds = EMRDataset(df, ctx_df)
MODEL_CONFIG["raw_concept_vocab_size"] = len(ds.rawconcept2id)
MODEL_CONFIG["concept_vocab_size"] = len(ds.concept2id)
MODEL_CONFIG["value_vocab_size"] = len(ds.value2id)
MODEL_CONFIG["vocab_size"] = len(ds.token2id)
MODEL_CONFIG["ctx_dim"] = ds.context_df.shape[1]
```

### 2. Train Model

```python
from transform_emr.train import run_two_phase_training
run_two_phase_training()
```

Model checkpoints and scaler are saved under `checkpoints/phase1/` and `checkpoints/phase2/`.
You can also split this part to it's components, running the prepare_data(), phase_one(), phase_two() seperatly,
but you'll need to adjust the imports. Use `train.py` structure for that.

### 3. Inference from the Model

```python
import joblib
from pathlib import Path
from pandas import pd

# Load data (verify you paths are properly defined)
temporal_df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
ctx_df = pd.read_csv(TEST_CTX_DATA_FILE, low_memory=False)

# You'll need to verify your temporal_df
# - Already has all the columns ['PatientID', 'ConceptName', 'Value', 'StartDateTime', 'EndDateTime']
# - Dates are in date-time (datetime64[ns]) format

# These should be updates from the training Dataset, or updated here manually:
MODEL_CONFIG["raw_concept_vocab_size"] = ...
MODEL_CONFIG["concept_vocab_size"] = ...
MODEL_CONFIG["value_vocab_size"] = ...
MODEL_CONFIG["vocab_size"] = ...
MODEL_CONFIG["ctx_dim"] = ...

# Load scaler
scaler_path = Path(EMBEDDER_CHECKPOINT).resolve.parent / "scaler.pkl"
scaler = joblib.load(scaler_path)

# Create the test dataset (cut input after k days for inference)
K=5 # Example for number of days you want as input
ds = EMRDataset(df, ctx_df, scaler=scaler, max_input_days=K)

# Load model
model = load_transformer()
model.eval()

results_df = infer_event_stream(model, ds, max_len=500)
```

This results_df will include both input events and generated events and will have these columns:
{"PatientID", "Step", "Token", "IsInput", "IsOutcome", "IsTerminal"}

You can analize the model's performance by comparing the input (full input) to the output (not directly)
 - Were all complications generated?
 - Were all complications generated on time? (use MEAL tokens to infer the time a model designated for an event)


### 4. Using as a module

You can perform local tests (not unit-tests) by activating the `.py` files, using the module as a package, as long as the file you are activating has __main__ section.

For example, run this from the root:
```bash
python -m transform_emr.train
```
---

## 🧪 Running Unit-Tests

Run all tests:

```bash
pytest tests/
```

📝 If you have **not trained the model yet**, skip inference-related tests:

```bash
pytest tests/test_dataset.py tests/test_embedder.py tests/test_train_pipeline.py
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
| `collate_emr()`     | Pads sequences and returns tensors|

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

The training loop and embedder design are adapted from Andrej Karpathy’s minimal GPT-2 implementation (https://github.com/karpathy/nanoGPT), with modifications for multi-embedding structure and k-step prediction loss.