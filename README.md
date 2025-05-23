# Event Prediction in EMRs

This repository implements a two-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and Transformer-based sequence modeling to predict or impute patient events over time.

This repo is part of an unpublished thesis and will be finalized post-submission. **Please do not reuse without permission**.

This model will be used on actual EMR data, stored in a closed environment. For that, it is organized as a package that can be installed:

```bash
event-prediction-in-diabetes-care/
│
├── transform_emr/                     # Core Python package
│   ├── config/                        # Configuration modules
│   │   ├── dataset_config.py
│   │   └── model_config.py
│   │
│   ├── dataset.py                     # Dataset, DataPreprocess and Tokenizer
│   ├── embedder.py                    # Embedding model (EMREmbedding) + training
│   ├── transformer.py                 # Transformer architecture (GPT) + training
│   ├── train.py                       # Full training pipeline (2-phase)
│   ├── inference.py                   # Inference pipeline
│   └── utils.py                       # Utility functions for the package (plots + loss penalties)
│
├── data/                              # External data folder (for synthetic or real EMR)
│   ├── generate_synthetic_data.ipynb  # A notebook that generates synthetic data similar in structure to original
│   ├── train/
│   └── test/
│
├── tests/                             # Unit and integration tests
│
├── .gitignore
├── requirements.txt
├── LICENCE
├── CITATION.cff
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
ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

print(f"[Pre-processing]: Building tokenizer...")
processor = DataProcessor(temporal_df, ctx_df, scaler=None)
temporal_df, ctx_df = processor.run()

tokenizer = EMRTokenizer.from_processed_df(temporal_df)
train_ds = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
MODEL_CONFIG['ctx_dim'] = train_ds.context_df.shape[1] # Dinamically updating shape
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
    import random
    import joblib
    from pathlib import Path
    from transform_emr.embedder import EMREmbedding
    from transform_emr.transformer import GPT
    from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
    from transform_emr.config.model_config import *

    # Load test data
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # Load tokenizer and scaler
    tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
    scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

    # Run preprocessing
    processor = DataProcessor(df, ctx_df, scaler=scaler, max_input_days=5)
    df, ctx_df = processor.run()

    patient_ids = df["PatientID"].unique()
    df_subset = df[df["PatientID"].isin(patient_ids)].copy()
    ctx_subset = ctx_df.loc[patient_ids].copy()

    # Create dataset
    dataset = EMRDataset(df_subset, ctx_subset, tokenizer=tokenizer)

    # Load models
    embedder, _, _, _, _ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _ = GPT.load(TRANSFORMER_CHECKPOINT, embedder=embedder)
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset, temperature=1.0)  # optional: adjust temperature
```

This results_df will include both input events and generated events and will have these columns:
{"PatientID", "Step", "Token", "IsInput", "IsOutcome", "IsTerminal", "TimeDelta", "TimePoint"}

You can analize the model's performance by comparing the input (`dataset.tokens_df`) to the output:
 - Were all complications generated?
 - Were all complications generated on time? (Set a forgiving boundry)


### 4. Using as a module

You can perform local tests (not unit-tests) by activating the `.py` files, using the module as a package, as long as the file you are activating has __main__ section.

For example, run this from the root:
```bash
python -m transform_emr.train

# Or

python -m transform_emr.inference

# Both modules have a __main__ activation to train / infer on a trained model 
```
---

## 🧪 Running Unit-Tests

Run all tests:

```bash
pytest tests/
```

📝 If you have **not trained a model yet**, skip inference-related tests:

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
| `DataProcessor`        | Performs all necessary data processing, from input data to tokens_df.  |
| `EMRTokenizer`        | Transforming a processed temporal_df into a tokenizer that can be saved and passed between objects for compatability.                     |
| `EMRDataset`        | Converts raw EMR tables into per-patient token sequences with relative time.                     |

| `collate_emr()`     | Pads sequences and returns tensors|

📌 **Why it matters:**  
Medical data varies in density and structure across patients. This dynamic preprocessing handles irregularity while preserving medically-relevant sequencing via `START/END` logic and relative timing.

---

### 2. **`embedder.py`** – EMR Representation Learning

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `Time2Vec`          | Learns periodic + trend encoding from inter-event durations.                                      |
| `EMREmbedding`      | Combines token, time, and patient context embeddings. Adds `[CTX]` token for global patient info. |
| `train_embedder()`  | Trains the embedding model with teacher-forced next-token prediction.                            |

⚙️ **Phase 1: Learning Events Representation**  
Phase 1 learns a robust, patient-aware representation of their event sequences. It isolates the core structure of patient timelines without being confounded by the autoregressive depth of Transformers.
The embedder uses:
- 4 levels of tokens - The event token is seperated to 4 hierarichal components to impose similarity between tokens of the same domain: `GLUCOSE` -> `GLUCOSE_TREND` -> `GLUCOSE_TREND_Inc` -> `GLUCOSE_TREND_Inc_Start`
- 2 levels of time - Delta T from the previous event, to predict local patterns, and ABS T from ADMISSION, to understand global patterns.

---

### 3. **`transformer.py`** – Causal Language Model over EMR Timelines

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `GPT`               | Transformer decoder stack over learned embeddings for next token prediction, with an additional head for delta_t prediction. Model inputs a trained embedder.                                               |
| `CausalSelfAttention` | Multi-head attention using causal mask to enforce chronology.                                 |
| `train_transformer()` | Complete training logic for the model using a BCE with multi-hot targets to account for EMR irregularities.                         |

⚙️ **Phase 2: Learning Sequence Dependencies**  
Once the EMR structure is captured, the transformer learns to model sequential dependencies in event progression:  
- What tends to follow a certain event?  
- How does timing affect outcomes?  
- How does patient context modulate the trajectory?

---

### 4. **`inference.py`** – Generating output from the model

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `get_token_embedding()` | Select a token and get it's embeddings based on an input embedder.                                 |
| `infer_event_stream()` | Generate predicted stream of events on an input dataset (Test).                         |

---

## ✅ Model Capabilities

- ✔️ **Handles irregular time-series data** using relative deltas and Time2Vec.
- ✔️ **Captures both short- and long-range dependencies** with deep transformer blocks.
- ✔️ **Supports variable-length patient histories** using custom collate and attention masks.
- ✔️ **Imputes and predicts** events in structured EMR timelines.

---

## 📚 Citation & Acknowledgments

This work builds on and adapts ideas from the following sources:

- **Time2Vec** (Kazemi et al., 2019):  
  The temporal embedding design is adapted from the Time2Vec formulation.  
  📄 *A. Kazemi, S. Ghamizi, A.-H. Karimi. "Time2Vec: Learning a Vector Representation of Time." NeurIPS 2019 Time Series Workshop.*  
  [arXiv:1907.05321](https://arxiv.org/abs/1907.05321)

- **nanoGPT** (Karpathy, 2023):  
  The training loop and transformer backbone are adapted from [nanoGPT](https://github.com/karpathy/nanoGPT),  
  with modifications for multi-stream EMR inputs, multiple embeddings, and a k-step prediction loss.
