# train_two_phase.py
# -----------------------------------------------------------
#   Two‑phase pre‑training driver for EMR language modelling
#   Phase‑1 : pre‑train Time2Vec‑based embedding
#   Phase‑2 : stack causal Transformer, keep embedder frozen
# -----------------------------------------------------------
import json, math, csv, argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---- your local modules -----------------------------------
from dataset     import EMRDataset, collate_emr          # <- exists in dataset.py
from embedding   import EMREmbedding
from transformer import EMRTransformer                  # <- you built this

# -----------------------------------------------------------
# utils
# -----------------------------------------------------------
class EarlyStopper:
    def __init__(self, patience=4, min_delta=0.):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.bad_epochs = 0
        self.stop       = False

    def step(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True        # new best
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stop = True
        return False

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, loss = model(**batch)
        total  += loss.item() * batch["targets"].numel()
        count  += batch["targets"].numel()
    return total / count

def save_ckpt(obj, optim, epoch, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch,
                "model_state": obj.state_dict(),
                "optim_state": optim.state_dict()}, path)

# -----------------------------------------------------------
# phase‑1  (pre‑train embedder)
# -----------------------------------------------------------
def phase1_embedder(cfg: Dict[str, Any], device) -> Path:
    # ------- data -------
    token2id = json.load(open(cfg["vocab_file"]))
    pad_id   = 0
    train_ds = EMRDataset(
        events_df      = cfg["train_events"],
        patient_ctx_df = cfg["patient_ctx"],
        numeric_concepts = cfg["numeric_concepts"],
        context_columns  = cfg["context_columns"],
        token2id = token2id,
        pad_id   = pad_id,
    )
    val_ds   = EMRDataset(
        events_df      = cfg["val_events"],
        patient_ctx_df = cfg["patient_ctx"],
        numeric_concepts = cfg["numeric_concepts"],
        context_columns  = cfg["context_columns"],
        token2id = token2id,
        pad_id   = pad_id,
    )
    train_ld = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                          collate_fn=lambda b: collate_emr(b, pad_id),
                          num_workers=cfg["num_workers"])
    val_ld   = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False,
                          collate_fn=lambda b: collate_emr(b, pad_id),
                          num_workers=cfg["num_workers"])

    # ------- model -------
    embedder = EMREmbedding(
        vocab_size   = len(token2id),
        embed_dim    = cfg["embed_dim"],
        time2vec_dim = cfg["time2vec_dim"],
        context_dim  = len(cfg["context_columns"]),
        pad_id       = pad_id
    ).to(device)

    decoder = nn.Linear(cfg["embed_dim"], len(token2id), bias=False).to(device)
    decoder.weight = embedder.token_embed.weight           # weight tying
    params = list(embedder.parameters()) + list(decoder.parameters())
    optim  = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-2)

    # ------- logging -------
    outdir = Path(cfg["out_dir"]) / "phase1"
    writer = SummaryWriter(outdir / "tb")
    csv_f  = open(outdir / "train_log.csv", "w", newline="")
    csv_wr = csv.writer(csv_f); csv_wr.writerow(["step", "split", "loss"])

    stopper, gstep = EarlyStopper(cfg["patience"], cfg["min_delta"]), 0

    # ------- training loop -------
    for epoch in range(cfg["epochs"]):
        embedder.train(); decoder.train()
        for batch in train_ld:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = decoder(embedder(batch["token_ids"],
                                      batch["time_deltas"],
                                      batch["context_vec"])[:, :-1])
            loss = F.cross_entropy(logits, batch["targets"], ignore_index=pad_id)
            loss.backward()
            optim.step(); optim.zero_grad()

            writer.add_scalar("train/loss", loss.item(), gstep)
            csv_wr.writerow([gstep, "train", loss.item()])
            gstep += 1

        # validate ----------------------------------------------------------
        embedder.eval(); decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_ld:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = decoder(embedder(batch["token_ids"],
                                          batch["time_deltas"],
                                          batch["context_vec"])[:, :-1])
                val_loss += F.cross_entropy(logits, batch["targets"],
                                            ignore_index=pad_id,
                                            reduction="sum").item()
        val_loss /= len(val_ds)
        writer.add_scalar("val/loss", val_loss, epoch)
        csv_wr.writerow([epoch, "val", val_loss])

        print(f"[PH1] epoch {epoch:02d}   val_loss={val_loss:.4f}")
        save_ckpt(embedder, optim, epoch, outdir / f"ckpt_epoch{epoch:02d}.pt")

        if stopper.step(val_loss):
            torch.save(embedder.state_dict(), outdir / "best.pt")
        if stopper.stop:
            break

    csv_f.close(); writer.close()
    return outdir / "best.pt"

# -----------------------------------------------------------
# phase‑2  (train transformer with embedder frozen)
# -----------------------------------------------------------
def phase2_transformer(cfg: Dict[str, Any], embedder_ckpt: Path, device):
    token2id = json.load(open(cfg["vocab_file"]))
    pad_id   = 0

    train_ds = EMRDataset(cfg["train_events"], cfg["patient_ctx"],
                          cfg["numeric_concepts"], cfg["context_columns"],
                          token2id, pad_id)
    val_ds   = EMRDataset(cfg["val_events"], cfg["patient_ctx"],
                          cfg["numeric_concepts"], cfg["context_columns"],
                          token2id, pad_id)
    train_ld = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                          collate_fn=lambda b: collate_emr(b, pad_id),
                          num_workers=cfg["num_workers"])
    val_ld   = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False,
                          collate_fn=lambda b: collate_emr(b, pad_id),
                          num_workers=cfg["num_workers"])

    # ------- model -------
    embedder = EMREmbedding(
        vocab_size   = len(token2id),
        embed_dim    = cfg["embed_dim"],
        time2vec_dim = cfg["time2vec_dim"],
        context_dim  = len(cfg["context_columns"]),
        pad_id       = pad_id
    )
    embedder.load_state_dict(torch.load(embedder_ckpt, map_location="cpu"))
    for p in embedder.parameters(): p.requires_grad = False   # keep frozen

    model = EMRTransformer(embedder,
                           n_layer = cfg["n_layer"],
                           n_head  = cfg["n_head"],
                           dropout = cfg["dropout"]).to(device)

    optim  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)

    # ------- logging -------
    outdir = Path(cfg["out_dir"]) / "phase2"
    writer = SummaryWriter(outdir / "tb")
    csv_f  = open(outdir / "train_log.csv", "w", newline="")
    csv_wr = csv.writer(csv_f); csv_wr.writerow(["step", "split", "loss"])

    stopper, gstep = EarlyStopper(cfg["patience"], cfg["min_delta"]), 0

    # ------- loop -------
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_ld:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, loss = model(**batch)          # model handles CE
            loss.backward()
            optim.step(); optim.zero_grad()

            writer.add_scalar("train/loss", loss.item(), gstep)
            csv_wr.writerow([gstep, "train", loss.item()])
            gstep += 1

        val_loss = evaluate(model, val_ld, device)
        writer.add_scalar("val/loss", val_loss, epoch)
        csv_wr.writerow([epoch, "val", val_loss])

        print(f"[PH2] epoch {epoch:02d}   val_loss={val_loss:.4f}")
        save_ckpt(model, optim, epoch, outdir / f"ckpt_epoch{epoch:02d}.pt")

        if stopper.step(val_loss):
            torch.save(model.state_dict(), outdir / "best.pt")
        if stopper.stop:
            break

    csv_f.close(); writer.close()

# -----------------------------------------------------------
# main
# -----------------------------------------------------------
def load_cfg(path: str) -> Dict[str, Any]:
    if path:
        with open(path, "r") as f:
            return json.load(f)
    # ---------- defaults ----------
    return {
        # file paths (replace with your own CSV / DF objects)
        "train_events":   "train_events.csv",
        "val_events":     "val_events.csv",
        "patient_ctx":    "patient_ctx.csv",
        "vocab_file":     "vocab.json",

        # EMR specifics
        "numeric_concepts": ["GLUCOSE", "WBC"],
        "context_columns":  ["age", "gender"],

        # model sizes
        "embed_dim":   256,
        "time2vec_dim":32,
        "n_layer":     8,
        "n_head":      8,
        "dropout":     0.1,

        # optimisation
        "bs":          8,
        "lr":          3e-4,
        "epochs":      20,
        "patience":    4,
        "min_delta":   1e-3,
        "num_workers": 4,

        # misc
        "out_dir":     "checkpoints",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two‑phase EMR LM pre‑training")
    parser.add_argument("--config", default=None,
                        help="JSON file with hyper‑parameters; "
                             "keys should match those in load_cfg()")
    args = parser.parse_args()
    cfg  = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("# === Phase 1 : embedder pre‑training ===")
    best_embedder = phase1_embedder(cfg, device)

    print("# === Phase 2 : transformer pre‑training (embedder frozen) ===")
    phase2_transformer(cfg, best_embedder, device)

    print("Finished.  Logs & checkpoints are under", cfg["out_dir"])
