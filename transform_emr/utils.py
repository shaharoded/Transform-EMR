"""
utils.py
==============

general util functions for the package
"""
import matplotlib.pyplot as plt
import torch
from collections import Counter


# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import meal2rank



def plot_losses(train_losses, val_losses):
    """
    Plot train vs. validation loss to inspect training quality.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‑entropy loss")
    plt.title("Training vs. validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_multi_hot_targets(position_ids, padding_idx, vocab_size, k):
    """
    For each timestep t, mark all tokens in [t+1, t+k] in a multi-hot vector.
    """
    B, T = position_ids.shape
    targets = torch.zeros((B, T, vocab_size), dtype=torch.float32, device=position_ids.device)

    for step in range(1, k + 1):
        if T - step <= 0:
            continue
        for b in range(B):
            for t in range(T - step):
                token = position_ids[b, t + step].item()
                if token != padding_idx:
                    targets[b, t, token] = 1.0
    return targets


def penalty_meal_order(predicted_tokens, id2token):
    """
    Penalizes incorrect meal ordering in a cyclic daily schedule.
    Expects meals to follow MEAL_BREAKFAST → MEAL_LUNCH → MEAL_DINNER → MEAL_NIGHT → MEAL_BREAKFAST.

    Args:
        predicted_tokens (Tensor): [B, T] tensor of predicted token IDs.
        id2token (dict): Mapping from token ID to token string.

    Returns:
        float: Total penalty score.
    """
    penalty = 0.0
    for b in range(predicted_tokens.size(0)):
        sequence = []
        for t in range(predicted_tokens.size(1)):
            tok = predicted_tokens[b, t].item()
            tok_str = id2token.get(tok)
            if tok_str in meal2rank:
                sequence.append(meal2rank[tok_str])

        if not sequence:
            continue

        # Compare each to the previous in cyclic order
        for i in range(1, len(sequence)):
            prev = sequence[i - 1]
            curr = sequence[i]
            # valid if curr == (prev + 1) % len(MEAL_ORDER)
            expected_next = (prev + 1) % len(meal2rank)
            if curr != expected_next:
                penalty += 1.0
    return penalty


def penalty_hallucinated_intervals(predicted_tokens, target_tokens, id2token, start_suffix="_START", end_suffix="_END"):
    """
    Penalizes:
      - Intervals predicted with invalid open/close order (e.g., END before START).
      - Incomplete or unmatched intervals that are not in the target.

    Args:
        predicted_tokens (Tensor): [B, T] tensor of predicted token IDs.
        target_tokens (Tensor):    [B, T] tensor of target token IDs.
        id2token (dict): Token ID -> string.
        start_suffix (str): Interval start suffix (default: "_START").
        end_suffix (str): Interval end suffix (default: "_END").

    Returns:
        float: total penalty
    """
    penalty = 0.0
    B, T = predicted_tokens.shape

    def extract_intervals(tokens):
        sequence = []
        for t in range(T):
            tok_str = id2token[tokens[t].item()]
            if tok_str.endswith(start_suffix):
                base = tok_str.replace(start_suffix, "")
                sequence.append((base, "START"))
            elif tok_str.endswith(end_suffix):
                base = tok_str.replace(end_suffix, "")
                sequence.append((base, "END"))
        return sequence

    def decompose(sequence):
        """
        Separates sequence into:
        - complete: properly paired START->END in order (not really important so not saved)
        - incomplete: unmatched START or END tokens
        """
        incomplete = []
        stack = []
        for item in sequence:
            base, kind = item
            if kind == "START":
                stack.append(base)
            elif kind == "END":
                if base in stack:
                    stack.remove(base)
                else:
                    incomplete.append((base, "END"))
        # Any leftover STARTs in stack are incomplete
        for base in stack:
            incomplete.append((base, "START"))
        return incomplete

    for b in range(B):
        pred_seq = extract_intervals(predicted_tokens[b])
        tgt_seq = extract_intervals(target_tokens[b])

        pred_incomplete = decompose(pred_seq)
        tgt_incomplete = decompose(tgt_seq)

        # Step 1: Drop all complete intervals from both
        # Step 2: Drop from pred_incomplete anything that's also in tgt_incomplete
        pred_filtered = Counter(pred_incomplete)
        tgt_incomplete_counter = Counter(tgt_incomplete)

        for k in tgt_incomplete_counter:
            if k in pred_filtered:
                # Remove min(pred, target) occurrences
                remove_count = min(pred_filtered[k], tgt_incomplete_counter[k])
                pred_filtered[k] -= remove_count
                if pred_filtered[k] == 0:
                    del pred_filtered[k]

        # Remaining in pred_filtered are hallucinated or unmatched intervals
        penalty += sum(pred_filtered.values())

    return penalty


def penalty_false_positives(predictions, targets, token_weights, important_token_ids, threshold=0.5):
    """
    Penalizes overgeneration of important tokens, scaled by their importance weights.

    Only considers tokens in `important_token_ids`. Penalizes if a token is predicted more times
    than it appears in the ground truth (multi-hot). This prevents false positives on critical concepts.

    Args:
        predictions (Tensor): [B, T, V] — raw logits or probabilities
        targets     (Tensor): [B, T, V] — ground-truth multi-hot labels
        token_weights (Tensor): [V] — per-token importance weights
        important_token_ids (Iterable[int]): list or set of token IDs to consider for penalty
        threshold (float): threshold above which a token is considered predicted

    Returns:
        float: penalty scalar
    """
    with torch.no_grad():
        pred_bin = (predictions > threshold).float()         # [B, T, V]
        false_pos = torch.clamp(pred_bin - targets, min=0.0) # [B, T, V]
        fp_counts = false_pos.sum(dim=(0, 1))                # [V]

        # Mask and weight only the important token IDs
        important_ids = torch.tensor(list(important_token_ids), device=predictions.device)
        penalties = fp_counts[important_ids] * token_weights[important_ids]

        return penalties.sum().item()