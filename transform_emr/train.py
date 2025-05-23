"""
pre-train.py
=====================
A two-phase transformer training process:
Phase‑1 : call embedding.train()  ------------>  pretrained_embedder.pt
Phase‑2 : GPT( pretrained_embedder, fine-tuned during training )  ->  best.pt
"""

from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr
from transform_emr.embedder import EMREmbedding, train_embedder
from transform_emr.transformer import GPT, train_transformer
from transform_emr.utils import *
from transform_emr.config.model_config import *
from transform_emr.config.dataset_config import TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE

def summarize_patient_data_split(train_ds, val_ds, train_ids, val_ids, tokenizer):
    """
    Prints summary statistics about your train/val split:
    - Patient counts
    - Record counts
    - Context shapes
    - Event count per patient (min/max/avg)
    - Token coverage (raw, concept, value, position)
    """

    print("✅ Data Split Summary")
    print(f"  - Train patients: {len(train_ids)}")
    print(f"  - Val patients:   {len(val_ids)}")

    print(f"  - Train records:  {len(train_ds.tokens_df):,}")
    print(f"  - Val records:    {len(val_ds.tokens_df):,}")

    # Per-patient record count stats
    train_counts = train_ds.tokens_df.groupby('PatientID').size()
    val_counts = val_ds.tokens_df.groupby('PatientID').size()

    print(f"\n📊 Train patient records:")
    print(f"  - Min:     {train_counts.min()}")
    print(f"  - Max:     {train_counts.max()}")
    print(f"  - Mean:    {train_counts.mean():.1f}")
    print(f"  - Median:  {train_counts.median()}")

    print(f"\n📊 Val patient records:")
    print(f"  - Min:     {val_counts.min()}")
    print(f"  - Max:     {val_counts.max()}")
    print(f"  - Mean:    {val_counts.mean():.1f}")
    print(f"  - Median:  {val_counts.median()}")

    # Token vocab sizes (from tokenizer)
    print(f"\n🧠 Vocabulary sizes:")
    print(f"  - Raw concepts:     {len(tokenizer.rawconcept2id):,}")
    print(f"  - Concepts:         {len(tokenizer.concept2id):,}")
    print(f"  - Concept+Value:    {len(tokenizer.value2id):,}")
    print(f"  - Full Tokens:      {len(tokenizer.token2id):,}")


def prepare_data():
    print(f"[Pre-processing]: Reading dataset...")
    temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

    if os.path.exists(os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        print(f"[Pre-processing]: Loading tokenizer from checkpoint...")
        tokenizer = EMRTokenizer.load()

        processor = DataProcessor(temporal_df, ctx_df, scaler=None)
        temporal_df, ctx_df = processor.run()

    else:
        print(f"[Pre-processing]: Building tokenizer...")
        processor = DataProcessor(temporal_df, ctx_df, scaler=None)
        temporal_df, ctx_df = processor.run()
        tokenizer = EMRTokenizer.from_processed_df(temporal_df)
        tokenizer.save()

    print(f"[Pre-processing]: Building dataset...")
    pids = temporal_df["PatientID"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)

    train_df, val_df = temporal_df[temporal_df.PatientID.isin(train_ids)].copy(), temporal_df[temporal_df.PatientID.isin(val_ids)].copy()
    train_ctx, val_ctx = ctx_df.loc[ctx_df.index.isin(train_ids)], ctx_df.loc[ctx_df.index.isin(val_ids)]

    train_ds = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_df, val_ctx, tokenizer=tokenizer)
    
    summarize_patient_data_split(train_ds, val_ds, train_ids, val_ids, tokenizer)   

    MODEL_CONFIG['ctx_dim'] = train_ds.context_df.shape[1] # Dinamically updating shape

    train_dl = DataLoader(train_ds, batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=True, collate_fn=collate_emr, num_workers=os.cpu_count())
    val_dl   = DataLoader(val_ds,  batch_size=TRAINING_SETTINGS.get('batch_size'), shuffle=False, collate_fn=collate_emr, num_workers=os.cpu_count())
    return train_dl, val_dl, tokenizer

def phase_one(embedder, train_dl, val_dl, resume=True):
    return train_embedder(
        embedder=embedder,
        train_loader=train_dl,
        val_loader=val_dl,
        resume=resume
    )

def phase_two(model, train_dl, val_dl, tune_embedder=True, resume=True):
    return train_transformer(
                        model=model, 
                        train_dl=train_dl, 
                        val_dl=val_dl, 
                        tune_embedder=tune_embedder, 
                        resume=resume, 
                        checkpoint_path=TRANSFORMER_CHECKPOINT
                    )


def run_two_phase_training():
    train_dl, val_dl, tokenizer = prepare_data()

    # --- Phase 1: Train or resume embedder ---
    ckpt_embedder_path = Path(EMBEDDER_CHECKPOINT).resolve().parent / "ckpt_last.pt"

    if ckpt_embedder_path.exists():
        embedder, _, _, _, _ = EMREmbedding.load(ckpt_embedder_path, tokenizer=tokenizer)
    else:
        embedder = EMREmbedding(
            tokenizer=tokenizer,
            ctx_dim=MODEL_CONFIG.get("ctx_dim"),
            time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
            embed_dim=MODEL_CONFIG.get("embed_dim")
        )

    embedder, _, _ = phase_one(embedder=embedder, train_dl=train_dl, val_dl=val_dl, resume=True)

    # --- Phase 2: Train or resume transformer ---
    ckpt_last_path = Path(TRANSFORMER_CHECKPOINT).resolve().parent / "ckpt_last.pt"

    if ckpt_last_path.exists():
        model, _, _, _, _ = GPT.load(ckpt_last_path, embedder=embedder)
    else:
        model = GPT(cfg=MODEL_CONFIG, embedder=embedder)

    phase_two(model=model, train_dl=train_dl, val_dl=val_dl, tune_embedder=True, resume=True)



if __name__ == "__main__":
    train_dl, val_dl, tokenizer = prepare_data()

    # --- Phase 1: Train or resume embedder ---
    ckpt_embedder_path = Path(EMBEDDER_CHECKPOINT).resolve().parent / "ckpt_last.pt"

    if ckpt_embedder_path.exists():
        embedder, _, _, _, _ = EMREmbedding.load(ckpt_embedder_path, tokenizer=tokenizer)
    else:
        embedder = EMREmbedding(
            tokenizer=tokenizer,
            ctx_dim=MODEL_CONFIG.get("ctx_dim"),
            time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
            embed_dim=MODEL_CONFIG.get("embed_dim")
        )

    embedder, _, _ = phase_one(embedder=embedder, train_dl=train_dl, val_dl=val_dl, resume=True)

    # --- Phase 2: Train or resume transformer ---
    ckpt_last_path = Path(TRANSFORMER_CHECKPOINT).resolve().parent / "ckpt_last.pt"

    if ckpt_last_path.exists():
        model, _, _, _, _ = GPT.load(ckpt_last_path, embedder=embedder)
    else:
        model = GPT(cfg=MODEL_CONFIG, embedder=embedder)

    phase_two(model=model, train_dl=train_dl, val_dl=val_dl, tune_embedder=True, resume=True)
