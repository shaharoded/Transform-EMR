# Event Prediction in EMRs

This repository implements a two-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and Transformer-based sequence modeling to predict or impute patient events over time.

This model is a part of my thesis and will be used on actual EMR data, stored in a closed environment.

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
| `_expand_tokens()`  | Generates CONCEPT_VALUE_(START|END) or single tokens from events. Tokenizing using (START|END) for time intervals allows to capture the length of an event (TIRP - a state or trend).                         |
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

## 🧪 Synthetic Data & Testing

The project includes a `data/` folder with synthetic EMR samples for testing architecture logic, convergence behavior, and debugging training scripts.

NOTE: This data is random, so you will not get a properly trained model out of it. It's just for reference.

---

## 🔧 Configuration

All hyperparameters and file paths are managed under:
- `config/model_config.py`
- `config/dataset_config.py`

---

## 🏁 Getting Started

```bash
# Phase 1+2: Run full pipeline with Transformer
python pre-train.py

# You can also train a stand-alone embedder using:
python embedding.py
```
Use Tensorboard or utils.plot_losses() to inspect learning curves.
---

## 🔍 Notes
Currently tested on PyTorch 2.1 with `torch.compile()` enabled.

Training logs and checkpoints are saved under `checkpoints/phase1/` and `checkpoints/phase2/`.

---

## 📚 Citation
Inspired by recent advancements in temporal deep learning, sequence modeling in healthcare (BEHRT, RETAIN, Med-BERT), and Time2Vec (Kazemi et al.).