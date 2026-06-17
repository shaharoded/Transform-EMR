"""
evaluation.py — Fixed post-training evaluation for EMR autoresearch.

DO NOT MODIFY — these metrics define the optimization target for each research round.
The agent should NOT edit this file. Improving these metrics is the goal.

Metrics (computed on the held-out test set, not the training validation split):

  Primary   — mean_auroc : mean per-complication AUROC from pooled episode-level AUC.
                           Higher is better. Random = 0.5, perfect = 1.0.
  Secondary — mean_auprc : mean per-complication AUPRC from the same evaluation.
                           Higher is better. Reflects precision at varying recall thresholds.
  Tertiary  — mean_mae_hours : mean onset-prediction error in hours.
                               Lower is better.

Evaluation protocol (mirrors evaluation.ipynb exactly):
  1. Load held-out test data (data/test/ — never seen during training).
  2. Re-process with the scaler fitted on the training pool.
  3. Build two datasets: full (for ground truth) and truncated (EVAL_INPUT_DAYS-day seed).
  4. Generate one autoregressive trajectory per patient from the truncated seed.
  5. Divide each trajectory into EVAL_WINDOW_HOURS windows.
  6. Label each window: 1 if any ground-truth episode falls within ±EVAL_GRACE_HOURS.
  7. Pool all (patient, window) pairs → single AUROC/AUPRC per complication.
  8. Report mean across all complications that pass MIN_POSITIVES threshold.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from joblib import load as joblib_load
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from intervene_ar.dataset import DataProcessor, EMRDataset
from intervene_ar.config.dataset_config import TAK_REPO_PATH, OUTCOME_RARE_THRESHOLD_PCT
from intervene_ar.inference import generate

# ---------------------------------------------------------------------------
# Fixed evaluation constants (do not change)
# ---------------------------------------------------------------------------

EVAL_INPUT_DAYS  = 2      # days of patient history used as generation seed
EVAL_WINDOW_HOURS = 24.0  # non-overlapping prediction window size
EVAL_GRACE_HOURS  = 24.0  # tolerance added to each window edge for positive labelling
EVAL_MAX_LEN      = 500   # max generated steps per patient
EVAL_TEMPERATURE  = 1.0   # sampling temperature (no top-k filtering)
EVAL_FULL_HORIZON_HOURS = 336.0  # cap per-patient eval horizon at 14 days (matches training/inference)
# Strict lower bound (hours) on positive-label events. Outcomes occurring in
# [0, FORECAST_CUTOFF_HOURS] are observed inside the input seed and are not
# part of the forecasting task — including them as positive labels inflates
# AUPRC trivially (Enc) or deflates it via unreachable positives (Ar). The
# STraTS / GRU-D preprocess already uses this convention; this restores
# label-definition parity across all four methods.
FORECAST_CUTOFF_HOURS = EVAL_INPUT_DAYS * 24.0

# Eval-time outcome support threshold = same 1% used at data-load time
# (OUTCOME_RARE_THRESHOLD_PCT in dataset_config). Outcomes that already passed
# train-set filtering can still be rarer in the held-out test set, so we
# re-check at eval time. Below this share of positive patients an outcome's
# AUROC/AUPRC is reported as NaN (still printed in per-outcome) and excluded
# from headline means.
EVAL_PREVALENCE_THRESHOLD = OUTCOME_RARE_THRESHOLD_PCT / 100.0  # fraction (0.01)

# Outcomes excluded from the AUROC/AUPRC/F1 evaluation entirely.
# RELEASE_EVENT is the negation of DEATH_EVENT in this cohort (essentially no
# censoring — every patient has either DEATH or RELEASE). Including both in
# the discrimination headline double-counts the same terminal-event ranking
# task. RELEASE stays in the LM vocab (so the model emits it and we get
# trajectory-length signal) and is reported via length_of_stay_mae instead.
AUC_EXCLUDE = ("RELEASE_EVENT",)


def _min_positives(n_patients, threshold=EVAL_PREVALENCE_THRESHOLD):
    """Minimum positive count for an outcome's AUC to be emitted (≥1)."""
    return max(1, int(round(threshold * n_patients)))


def length_of_stay_mae(risk_df, gt_episodes, release_token="RELEASE_EVENT"):
    """
    Purpose: Length-of-stay regression MAE — replaces RELEASE peak-MAE.
    Method:  For each patient that was released in GT:
               GT_LoS  = earliest GT RELEASE timepoint (admission anchor = 0).
               Pred_LoS = last timestamp in patient's full sequence
                         (input + generated). Captures admission → end of
                         model-emitted trajectory.
             MAE = mean |Pred_LoS − GT_LoS| over released patients.

             Distinct from peak-MAE on RELEASE: peak-MAE asks "when does the
             model's risk-curve peak vs the nearest GT RELEASE token?" — a
             risk-curve-shape metric. LoS asks "did the model predict the
             right discharge timing?" — a trajectory-length regression.

    Args:
        risk_df (pd.DataFrame): generate() output with collect_risk_scores=True.
        gt_episodes (dict): {pid: {outcome: [t1, ...]}} from extract_ground_truth_episodes.
        release_token (str): token marking discharge.

    Returns:
        dict: mae_hours, median_hours, p90_hours, n_patients, gt_mean_hours, pred_mean_hours.
    """
    errors = []
    gt_vals = []
    pred_vals = []
    for pid, sub in risk_df.groupby("PatientId"):
        gt_releases = gt_episodes.get(pid, {}).get(release_token, [])
        if not gt_releases:
            continue
        gt_los = float(min(gt_releases))  # earliest GT release
        pred_los = float(sub["TimePoint"].max())  # admission→end of (input+generated)
        errors.append(abs(pred_los - gt_los))
        gt_vals.append(gt_los)
        pred_vals.append(pred_los)
    if not errors:
        return {
            "mae_hours":      float("nan"),
            "median_hours":   float("nan"),
            "p90_hours":      float("nan"),
            "n_patients":     0,
            "gt_mean_hours":  float("nan"),
            "pred_mean_hours": float("nan"),
        }
    errs = np.asarray(errors)
    return {
        "mae_hours":      float(errs.mean()),
        "median_hours":   float(np.median(errs)),
        "p90_hours":      float(np.percentile(errs, 90)),
        "n_patients":     int(len(errs)),
        "gt_mean_hours":  float(np.mean(gt_vals)),
        "pred_mean_hours": float(np.mean(pred_vals)),
    }


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(eval_ds, outcome_names,
                         min_event_time_hours=FORECAST_CUTOFF_HOURS):
    """
    Purpose: Build per-patient first-occurrence ground truth for each outcome.
    Method: Scans each patient's full (untruncated) token sequence; only
            collects occurrences strictly after `min_event_time_hours` (default
            FORECAST_CUTOFF_HOURS, i.e. the input seed window). Excluding
            in-seed events matches the STraTS / GRU-D preprocess label
            convention and stops trivial positives from inflating per-outcome
            n_pos counts.

    Args:
        eval_ds              (EMRDataset): Full (untruncated) test dataset.
        outcome_names        (list[str]): Outcome token strings to collect.
        min_event_time_hours (float):      strict lower bound; events at or
                                          before this time are ignored.

    Returns:
        dict: {patient_id: {outcome_name: first_time_hours or np.inf}}
    """
    outcome_set = set(outcome_names)
    tok_col     = "PositionToken" if "PositionToken" in next(iter(eval_ds.patient_groups.values())).columns else "Token"
    gt = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        patient_gt = {n: np.inf for n in outcome_names}
        for _, row in df.iterrows():
            tok = row[tok_col]
            if tok in outcome_set:
                t = row["TimePoint"]
                if t > min_event_time_hours and t < patient_gt[tok]:
                    patient_gt[tok] = t
        gt[pid] = patient_gt
    return gt


def compute_gen_stats(risk_df, patient_horizons=None):
    """
    Purpose: Honest diagnostics for the trajectory-collapse failure mode.
    Method:  Compute per-patient stats from the generated rows only (IsInput==0).
             When patient_horizons is provided, also compute the length-MAE between
             generated trajectory span and per-patient ground-truth horizon.

    Args:
        risk_df (pd.DataFrame): Output of generate() with collect_risk_scores=True.
        patient_horizons (dict, optional): {pid: horizon_hours} from extract_patient_horizons.

    Returns:
        dict: gen_median_steps, gen_mean_steps, gen_p90_steps, gen_max_steps,
              gen_median_hours, gen_mean_hours, gen_p90_hours, gen_max_hours,
              gen_frac_terminal_first24h, gen_n_with_terminal, gen_length_mae_hrs.
    """
    stats = {"gen_n_patients": int(risk_df["PatientId"].nunique())}

    gen_df = risk_df[risk_df["IsInput"] == 0]
    if len(gen_df) == 0:
        return stats

    per_pat_steps = gen_df.groupby("PatientId").size()
    span          = (gen_df.groupby("PatientId")["TimePoint"].max()
                     - gen_df.groupby("PatientId")["TimePoint"].min())
    seed_end      = gen_df.groupby("PatientId")["TimePoint"].min()

    # First-terminal time per patient (only patients that emitted one).
    term_df = gen_df[gen_df["IsTerminal"] == 1]
    if len(term_df):
        term       = term_df.groupby("PatientId")["TimePoint"].min()
        within24   = (term - seed_end.loc[term.index]).lt(24.0)
        n_terminal = int(len(term))
        frac_early = float(within24.mean())
    else:
        n_terminal = 0
        frac_early = 0.0

    stats.update({
        "gen_median_steps":          float(per_pat_steps.median()),
        "gen_mean_steps":            float(per_pat_steps.mean()),
        "gen_p90_steps":             float(per_pat_steps.quantile(0.9)),
        "gen_max_steps":             int(per_pat_steps.max()),
        "gen_median_hours":          float(span.median()),
        "gen_mean_hours":            float(span.mean()),
        "gen_p90_hours":             float(span.quantile(0.9)),
        "gen_max_hours":             float(span.max()),
        "gen_n_with_terminal":       n_terminal,
        "gen_frac_terminal_first24h": frac_early,
    })

    # Length-MAE vs GT horizon, plus GT length statistics for the agent to read
    # the ratio "how much of the patient's true admission did the model cover?"
    # at a glance.
    if patient_horizons:
        diffs = []
        gt_spans = []
        for pid, s in span.items():
            if pid not in patient_horizons:
                continue
            gt_span = max(0.0, patient_horizons[pid] - seed_end.loc[pid])
            gt_spans.append(gt_span)
            diffs.append(abs(float(s) - gt_span))
        if diffs:
            stats["gen_length_mae_hrs"] = float(np.mean(diffs))
        if gt_spans:
            gt_arr = np.asarray(gt_spans, dtype=float)
            stats["gt_median_hours"]    = float(np.median(gt_arr))
            stats["gt_mean_hours"]      = float(gt_arr.mean())
            stats["gt_p90_hours"]       = float(np.percentile(gt_arr, 90))
            # Ratios — primary trajectory-collapse summary metric. 1.0 = generation
            # spans the patient's true horizon; 0.0 = collapsed to immediate terminal.
            gt_median = stats["gt_median_hours"]
            gt_mean   = stats["gt_mean_hours"]
            stats["gen_to_gt_ratio_median"] = (float(span.median()) / gt_median) if gt_median > 0 else 0.0
            stats["gen_to_gt_ratio_mean"]   = (float(span.mean())   / gt_mean)   if gt_mean   > 0 else 0.0

    return stats


def extract_patient_horizons(eval_ds, full_horizon_hours=EVAL_FULL_HORIZON_HOURS):
    """
    Purpose: Per-patient evaluation horizon = min(last event timepoint, full_horizon_hours).
    Method: Read the maximum TimePoint from each patient's untruncated sequence; cap at
            the training trajectory horizon so we never evaluate past in-distribution time.

    Args:
        eval_ds (EMRDataset): Full (untruncated) dataset — same one used for ground-truth.
        full_horizon_hours (float): Hard cap (default 336 h = 14 days, matches inference).

    Returns:
        dict: {patient_id: horizon_hours}
    """
    out = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        last_t = float(df["TimePoint"].max()) if len(df) else 0.0
        out[pid] = min(last_t, full_horizon_hours)
    return out


def extract_ground_truth_episodes(eval_ds, outcome_names,
                                  min_event_time_hours=FORECAST_CUTOFF_HOURS):
    """
    Purpose: Build per-patient all-occurrence ground truth (list of times) for each outcome.
    Method: Scans each patient's full (untruncated) token sequence; collects
            only occurrences strictly after `min_event_time_hours` (default
            FORECAST_CUTOFF_HOURS, i.e. the input seed window). Excluding
            in-seed events matches the STraTS / GRU-D preprocess label
            convention so per-outcome positives reflect the forecasting task,
            not events visible in the model input.

    Args:
        eval_ds              (EMRDataset): Full (untruncated) test dataset.
        outcome_names        (list[str]): Outcome token strings to collect.
        min_event_time_hours (float):      strict lower bound; events at or
                                          before this time are ignored.

    Returns:
        dict: {patient_id: {outcome_name: [t1, t2, ...]}}  (empty list if never occurred)
    """
    outcome_set = set(outcome_names)
    tok_col     = "PositionToken" if "PositionToken" in next(iter(eval_ds.patient_groups.values())).columns else "Token"
    gt = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        patient_gt = {n: [] for n in outcome_names}
        for _, row in df.iterrows():
            tok = row[tok_col]
            if tok in outcome_set and row["TimePoint"] > min_event_time_hours:
                patient_gt[tok].append(row["TimePoint"])
        gt[pid] = patient_gt
    return gt


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def pooled_episode_auc(risk_df, gt_labels_episodes, outcome_names,
                        window_hours=EVAL_WINDOW_HOURS,
                        grace_hours=EVAL_GRACE_HOURS,
                        min_positives=None,
                        patient_horizons=None):
    """
    Purpose: Compute episode-level AUROC and AUPRC pooled across all (patient, window) pairs.
    Method: Build a per-patient window grid from the global earliest generated step time
            (t_start) to each patient's evaluation horizon. For each window:
              score = max P_<outcome> over generated tokens that fall inside the window
                      (0.0 when the model produced no tokens in that window — i.e. the
                      autoregressive trajectory had already terminated by then).
              label = 1 if any ground-truth episode of that outcome falls in
                      [win_start - grace_hours, win_end + grace_hours].
            This penalises the model for failing to predict outcomes that occur after it
            stopped generating: those windows become positive labels scored at zero.

    Args:
        risk_df (pd.DataFrame): Output of generate() with collect_risk_scores=True.
        gt_labels_episodes (dict): {pid: {outcome: [t1, t2, ...]}} all episode times in hours.
        outcome_names (list[str]): Outcome names to evaluate.
        window_hours (float): Duration of each evaluation window in hours.
        grace_hours (float): Extra tolerance added to each window edge for positive labelling.
        min_positives (int): Skip outcome if fewer than this many positive windows exist.
        patient_horizons (dict, optional): {pid: horizon_hours} from extract_patient_horizons.
            When provided, every patient is evaluated to its real horizon (capped at
            EVAL_FULL_HORIZON_HOURS) regardless of where generation stopped. When None,
            falls back to the patient's last generated step time (legacy behaviour).

    Returns:
        pd.DataFrame: Indexed by outcome, columns: auroc, auprc, n_pos_windows, n_neg_windows.
    """
    import math

    gen_df = risk_df[risk_df["IsInput"] == 0].copy()
    p_cols = [f"P_{n}" for n in outcome_names]
    if len(gen_df) == 0:
        return pd.DataFrame()

    if min_positives is None:
        min_positives = _min_positives(risk_df["PatientId"].nunique())

    t_start = float(gen_df["TimePoint"].min())

    # Per-patient horizon: caller-supplied or fall back to last generated step.
    if patient_horizons is None:
        patient_horizons = {pid: float(sub["TimePoint"].max())
                            for pid, sub in gen_df.groupby("PatientId")}

    # Group generated rows by patient once.
    gen_by_pid = {pid: sub for pid, sub in gen_df.groupby("PatientId")}

    # Build the window grid for every patient up to their horizon.
    rows = []
    for pid, horizon in patient_horizons.items():
        if horizon <= t_start:
            continue
        n_windows = max(1, int(math.ceil((horizon - t_start) / window_hours)))
        pat_gen = gen_by_pid.get(pid)
        for k in range(n_windows):
            ws = t_start + k * window_hours
            we = ws + window_hours
            row = {"PatientId": pid, "_t_start": ws, "_t_end": we}
            if pat_gen is not None:
                in_win = pat_gen[(pat_gen["TimePoint"] >= ws) & (pat_gen["TimePoint"] < we)]
                if len(in_win) > 0:
                    for pcol in p_cols:
                        row[pcol] = float(in_win[pcol].max())
                else:
                    for pcol in p_cols:
                        row[pcol] = 0.0
            else:
                for pcol in p_cols:
                    row[pcol] = 0.0
            rows.append(row)

    peak = pd.DataFrame(rows)

    # Score / label loop (identical to before, just over the extended window grid).
    result_rows = []
    for name in outcome_names:
        pcol   = f"P_{name}"
        scores, labels = [], []
        for _, row in peak.iterrows():
            pid      = row["PatientId"]
            t_lo     = row["_t_start"] - grace_hours
            t_hi     = row["_t_end"]   + grace_hours
            episodes = gt_labels_episodes.get(pid, {}).get(name, [])
            label    = int(any(t_lo <= ep <= t_hi for ep in episodes))
            scores.append(row[pcol])
            labels.append(label)

        labels = np.array(labels)
        scores = np.array(scores)
        n_pos  = int(labels.sum())
        n_neg  = int((1 - labels).sum())

        if n_pos < min_positives:
            result_rows.append({"outcome": name, "auroc": np.nan, "auprc": np.nan,
                                "n_pos_windows": n_pos, "n_neg_windows": n_neg})
            continue

        result_rows.append({
            "outcome":       name,
            "auroc":         roc_auc_score(labels, scores),
            "auprc":         average_precision_score(labels, scores),
            "n_pos_windows": n_pos,
            "n_neg_windows": n_neg,
        })

    return pd.DataFrame(result_rows).set_index("outcome").sort_values("auroc", ascending=False)


def pooled_auc_across_horizons(risk_df, gt_labels_episodes, outcome_names,
                                eval_ds_full,
                                horizon_caps_hrs=tuple(range(24, 337, 24)),
                                window_hours=EVAL_WINDOW_HOURS,
                                grace_hours=EVAL_GRACE_HOURS,
                                min_positives=None):
    """
    Purpose: Compute pooled_episode_auc at multiple per-patient horizon caps so
             the agent can read off a horizon curve in a single eval pass —
             a cheap "next-48h", a medium "first-week", and the full 14-day
             extension, all from the same generated risk_df.
    Method:  Build a patient_horizons dict per cap = min(GT_last_event, cap),
             call pooled_episode_auc for each, stack the results as a long
             DataFrame indexed by (horizon_cap_hrs, outcome).

    Args:
        risk_df (pd.DataFrame): generate() output with collect_risk_scores=True.
        gt_labels_episodes (dict): from extract_ground_truth_episodes.
        outcome_names (list[str]): canonical outcome names.
        eval_ds_full (EMRDataset): untruncated test dataset for horizon extraction.
        horizon_caps_hrs (tuple): per-patient horizon caps to evaluate at.
        window_hours, grace_hours, min_positives: forwarded to pooled_episode_auc.

    Returns:
        pd.DataFrame: columns horizon_cap_hrs, outcome, auroc, auprc, n_pos, n_neg.
    """
    rows = []
    for cap in horizon_caps_hrs:
        horizons = extract_patient_horizons(eval_ds_full, full_horizon_hours=float(cap))
        tbl = pooled_episode_auc(risk_df, gt_labels_episodes, outcome_names,
                                  window_hours=window_hours, grace_hours=grace_hours,
                                  min_positives=min_positives,
                                  patient_horizons=horizons)
        for outcome, row in tbl.iterrows():
            rows.append({
                "horizon_cap_hrs": int(cap),
                "outcome":         outcome,
                "auroc":           row["auroc"],
                "auprc":           row["auprc"],
                "n_pos":           int(row["n_pos_windows"]),
                "n_neg":           int(row["n_neg_windows"]),
            })
    return pd.DataFrame(rows)


def time_accuracy(risk_df, gt_labels, outcome_names):
    """
    Purpose: Compute mean absolute error between predicted and actual complication onset time.
    Method: For each patient where a complication occurred, finds the generated step with peak
            outcome-head probability and measures its distance from the ground-truth FIRST time.

    Args:
        risk_df (pd.DataFrame): Output of generate() with collect_risk_scores=True.
        gt_labels (dict): {pid: {outcome: first_time_hours or np.inf}}.
        outcome_names (list[str]): Outcome names to evaluate.

    Returns:
        pd.DataFrame: Indexed by outcome, columns: mae_hours, n_patients.
    """
    gen_df = risk_df[risk_df["IsInput"] == 0].copy()
    p_cols = [f"P_{n}" for n in outcome_names]
    idxmax = gen_df.groupby("PatientId")[p_cols].idxmax()

    rows = []
    for name in outcome_names:
        pcol   = f"P_{name}"
        pred_t = gen_df.loc[idxmax[pcol].dropna().astype(int), ["PatientId", "TimePoint"]]
        pred_t = pred_t.set_index("PatientId")["TimePoint"]

        errors = []
        for pid, pt in pred_t.items():
            gt_t = gt_labels.get(pid, {}).get(name, np.inf)
            if gt_t < np.inf:
                errors.append(abs(pt - gt_t))

        rows.append({
            "outcome":    name,
            "mae_hours":  np.mean(errors) if errors else np.nan,
            "n_patients": len(errors),
        })

    return pd.DataFrame(rows).set_index("outcome").sort_values("mae_hours")


def time_accuracy_nearest(risk_df, gt_episodes, outcome_names):
    """
    Purpose: MAE between the model's peak-risk moment and the NEAREST ground-truth
             occurrence (not just the first). Fairer when complications recur:
             argmax may catch the more prominent of two correct hits.
    Method:  For each (patient, outcome), find t_peak = argmax_t P_outcome(t) in the
             generated portion, then mae = min_{t_gt in episodes} |t_peak − t_gt|.
             Patients with no GT occurrence of that outcome are skipped.

    Args:
        risk_df (pd.DataFrame): Output of generate() with collect_risk_scores=True.
        gt_episodes (dict): {pid: {outcome: [t1, t2, ...]}} all occurrence times.
        outcome_names (list[str]): Outcome names to evaluate.

    Returns:
        pd.DataFrame: Indexed by outcome, columns: mae_hours, n_patients.
    """
    gen_df = risk_df[risk_df["IsInput"] == 0].copy()
    if len(gen_df) == 0:
        return pd.DataFrame()
    p_cols = [f"P_{n}" for n in outcome_names]
    idxmax = gen_df.groupby("PatientId")[p_cols].idxmax()

    rows = []
    for name in outcome_names:
        pcol   = f"P_{name}"
        pred_t = gen_df.loc[idxmax[pcol].dropna().astype(int), ["PatientId", "TimePoint"]]
        pred_t = pred_t.set_index("PatientId")["TimePoint"]

        errors = []
        for pid, pt in pred_t.items():
            episodes = gt_episodes.get(pid, {}).get(name, [])
            if not episodes:
                continue
            # Distance to nearest GT occurrence.
            errors.append(min(abs(pt - t_gt) for t_gt in episodes))

        rows.append({
            "outcome":    name,
            "mae_hours":  float(np.mean(errors)) if errors else np.nan,
            "n_patients": len(errors),
        })

    return pd.DataFrame(rows).set_index("outcome").sort_values("mae_hours")


def per_patient_max_auc(risk_df, gt_episodes, outcome_names, min_positives=None):
    """
    Purpose: Patient-level peak-detector AUC (new headline framing).
    Method:  For each (patient, outcome):
               score = max P_outcome(t) over all generated positions (IsInput==0).
                       Patients that generated no tokens contribute score = 0.
               label = 1 iff the outcome occurred at any point in the GT trajectory.
             AUROC/AUPRC computed once per outcome over all patients.

             This replaces the per-(patient, window) pooling used by
             pooled_episode_auc: rare-outcome AUCs are far more stable here
             because each outcome reduces to a single binary classification
             with n_patient positives vs negatives — no window-count noise
             amplification.

    Args:
        risk_df (pd.DataFrame): generate() output with collect_risk_scores=True.
        gt_episodes (dict): {pid: {outcome: [t1, t2, ...]}}.
        outcome_names (list[str]): outcomes to score.
        min_positives (int, optional): minimum positive patients to emit an AUC.
            Defaults to round(EVAL_PREVALENCE_THRESHOLD * n_patients), so the
            same 1 % support rule the data pipeline uses applies here too.

    Returns:
        pd.DataFrame: indexed by outcome, columns:
            auroc, auprc, max_f1, max_f1_threshold, f1_at_0_5, n_pos, n_neg, prevalence
    """
    gen_df = risk_df[risk_df["IsInput"] == 0]
    p_cols = [f"P_{n}" for n in outcome_names]
    all_pids = list(risk_df["PatientId"].unique())
    n_patients = len(all_pids)
    if min_positives is None:
        min_positives = _min_positives(n_patients)

    # Per-patient max score per outcome. Patients with no generated rows → 0.
    max_per_patient = {pid: {c: 0.0 for c in p_cols} for pid in all_pids}
    if len(gen_df):
        grouped = gen_df.groupby("PatientId")[p_cols].max()
        for pid, row in grouped.iterrows():
            for c in p_cols:
                max_per_patient[pid][c] = float(row[c])

    rows = []
    for name in outcome_names:
        pcol = f"P_{name}"
        scores, labels = [], []
        for pid in all_pids:
            scores.append(max_per_patient[pid][pcol])
            labels.append(int(len(gt_episodes.get(pid, {}).get(name, [])) > 0))
        labels = np.array(labels)
        scores = np.array(scores)
        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())
        prevalence = n_pos / max(1, n_pos + n_neg)

        if n_pos < min_positives or n_neg < min_positives:
            rows.append({"outcome": name, "auroc": np.nan, "auprc": np.nan,
                         "max_f1": np.nan, "max_f1_threshold": np.nan,
                         "f1_at_0_5": np.nan,
                         "n_pos": n_pos, "n_neg": n_neg, "prevalence": prevalence})
            continue

        # Max-F1 by sweeping the precision-recall curve.
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        # precisions/recalls have length len(thresholds)+1; last point is (recall=0, prec=1).
        f1s = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12),
            0.0,
        )
        best_idx = int(np.argmax(f1s))
        max_f1 = float(f1s[best_idx])
        # thresholds has one fewer element than precisions/recalls; cap at last threshold.
        if best_idx < len(thresholds):
            max_f1_thr = float(thresholds[best_idx])
        else:
            max_f1_thr = float(thresholds[-1]) if len(thresholds) else 0.5

        # F1 at threshold 0.5.
        preds_05 = (scores >= 0.5).astype(int)
        tp = int(((preds_05 == 1) & (labels == 1)).sum())
        fp = int(((preds_05 == 1) & (labels == 0)).sum())
        fn = int(((preds_05 == 0) & (labels == 1)).sum())
        prec_05 = tp / max(tp + fp, 1)
        rec_05  = tp / max(tp + fn, 1)
        if prec_05 + rec_05 > 0:
            f1_at_0_5 = 2 * prec_05 * rec_05 / (prec_05 + rec_05)
        else:
            f1_at_0_5 = 0.0

        rows.append({
            "outcome":          name,
            "auroc":            float(roc_auc_score(labels, scores)),
            "auprc":            float(average_precision_score(labels, scores)),
            "max_f1":           max_f1,
            "max_f1_threshold": max_f1_thr,
            "f1_at_0_5":        float(f1_at_0_5),
            "n_pos":            n_pos,
            "n_neg":            n_neg,
            "prevalence":       prevalence,
        })

    return pd.DataFrame(rows).set_index("outcome").sort_values("auroc", ascending=False)


def weighted_mean_auc(auc_table, by="n_pos"):
    """
    Purpose: Support-weighted mean AUROC/AUPRC across outcomes.
    Method:  Σ(w_o · AUC_o) / Σ(w_o) over outcomes with non-NaN AUC.
             Weight defaults to n_pos so rare outcomes contribute less.

    Args:
        auc_table (pd.DataFrame): per-outcome table with columns
            auroc, auprc, n_pos (e.g. from per_patient_max_auc).
        by (str): weight column ("n_pos" or "prevalence").

    Returns:
        dict: {"auroc_weighted", "auprc_weighted", "auroc_simple",
               "auprc_simple", "n_outcomes_used"}.
    """
    tbl = auc_table.dropna(subset=["auroc"])
    if len(tbl) == 0:
        nan = float("nan")
        return {"auroc_weighted": nan, "auprc_weighted": nan,
                "auroc_simple":   nan, "auprc_simple":   nan,
                "max_f1_weighted": nan, "max_f1_simple": nan,
                "f1_at_0_5_weighted": nan, "f1_at_0_5_simple": nan,
                "n_outcomes_used": 0}
    w = tbl[by].astype(float).values
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    # F1 columns may not exist on legacy callers; default to nan-safe sums.
    has_max_f1   = "max_f1"    in tbl.columns
    has_f1_05    = "f1_at_0_5" in tbl.columns
    return {
        "auroc_weighted":      float((tbl["auroc"].values * w).sum()),
        "auprc_weighted":      float((tbl["auprc"].values * w).sum()),
        "auroc_simple":        float(tbl["auroc"].mean()),
        "auprc_simple":        float(tbl["auprc"].mean()),
        "max_f1_weighted":     float((tbl["max_f1"].values    * w).sum()) if has_max_f1 else float("nan"),
        "max_f1_simple":       float(tbl["max_f1"].mean())                 if has_max_f1 else float("nan"),
        "f1_at_0_5_weighted":  float((tbl["f1_at_0_5"].values * w).sum()) if has_f1_05  else float("nan"),
        "f1_at_0_5_simple":    float(tbl["f1_at_0_5"].mean())              if has_f1_05  else float("nan"),
        "n_outcomes_used":     int(len(tbl)),
    }


# ---------------------------------------------------------------------------
# Main evaluation entry point (called by api.py)
# ---------------------------------------------------------------------------

def evaluate_on_test_set(model, tokenizer, test_temporal_raw, test_ctx_raw, scaler, checkpoint_dir):
    """
    Purpose: Full post-training evaluation on the held-out test set.
    Method: Re-processes the raw test data twice — once untruncated (for ground truth) and
            once with EVAL_INPUT_DAYS truncation (for generation seed) — then generates
            risk curves and computes episode-level AUROC/AUPRC and onset-time MAE.

    Args:
        model: Trained InterveneGPT model (best available checkpoint, already loaded).
        tokenizer (EMRTokenizer): Fitted tokenizer (same as used during training).
        test_temporal_raw (pd.DataFrame): Raw (unprocessed) test temporal events.
        test_ctx_raw (pd.DataFrame): Raw (unprocessed) test context features.
        scaler: Fitted StandardScaler from training (loaded from checkpoints/scaler.pkl).
        checkpoint_dir (str): Path to checkpoints directory.

    Returns:
        dict with keys:
            mean_auroc (float)      : mean per-complication AUROC  [primary, higher is better]
            mean_auprc (float)      : mean per-complication AUPRC  [secondary, higher is better]
            mean_mae_hours (float)  : mean onset-prediction MAE    [tertiary, lower is better]
            auc_table (pd.DataFrame): per-outcome AUROC/AUPRC/n_windows table
            mae_table (pd.DataFrame): per-outcome MAE/n_patients table
    """
    # -- Full dataset (untruncated, for ground truth extraction) --
    print("[Eval] Processing full test sequences (ground truth)...")
    full_proc = DataProcessor(
        test_temporal_raw.copy(), test_ctx_raw.copy(),
        scaler=scaler,
        tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=checkpoint_dir,
    )
    full_temporal_df, full_ctx_df = full_proc.run()
    eval_ds_full = EMRDataset(full_temporal_df, full_ctx_df, tokenizer=tokenizer)

    # -- Truncated dataset (EVAL_INPUT_DAYS seed for generation) --
    print(f"[Eval] Processing truncated test sequences ({EVAL_INPUT_DAYS}-day input)...")
    trunc_proc = DataProcessor(
        test_temporal_raw.copy(), test_ctx_raw.copy(),
        scaler=scaler,
        tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=checkpoint_dir,
        max_input_days=EVAL_INPUT_DAYS,
    )
    trunc_temporal_df, trunc_ctx_df = trunc_proc.run()
    eval_ds_input = EMRDataset(trunc_temporal_df, trunc_ctx_df, tokenizer=tokenizer)

    # -- Generate risk curves --
    print("[Eval] Generating risk curves...")
    model.eval()
    risk_df = generate(
        model, eval_ds_input,
        max_len=EVAL_MAX_LEN,
        temperature=EVAL_TEMPERATURE,
        top_k=None,
        rep_decay=0.6,
        collect_risk_scores=True,
    )
    print(f"[Eval] Generated {len(risk_df)} rows for {risk_df['PatientId'].nunique()} patients.")

    outcome_names = model.outcome_names

    # -- Extract ground truth + per-patient evaluation horizons --
    gt_first         = extract_ground_truth(eval_ds_full, outcome_names)
    gt_episodes      = extract_ground_truth_episodes(eval_ds_full, outcome_names)
    patient_horizons = extract_patient_horizons(eval_ds_full)
    horizons_arr     = np.array(list(patient_horizons.values()), dtype=float)
    print(f"[Eval] Patient horizons (h): median={np.median(horizons_arr):.1f}, "
          f"mean={horizons_arr.mean():.1f}, p90={np.percentile(horizons_arr, 90):.1f}, "
          f"max={horizons_arr.max():.1f}")

    # -- Compute metrics --
    print("[Eval] Computing patient-level AUC, episode-level AUC, time accuracy...")
    # AUC/F1 outcomes EXCLUDE RELEASE_EVENT (it's the negation of DEATH; double-
    # counts the same ranking task). RELEASE is reported via length_of_stay_mae
    # below and stays in the LM vocab so the model still emits it.
    auc_outcome_names = [n for n in outcome_names if n not in AUC_EXCLUDE]
    print(f"[Eval] AUC/F1 computed over {len(auc_outcome_names)} outcomes "
          f"(excluded from AUROC headline: {list(AUC_EXCLUDE)}).")
    # NEW HEADLINE — per-patient peak-detector AUC. Each (patient, outcome)
    # contributes one (max_P, label) pair; far more stable than per-window.
    patient_auc_table = per_patient_max_auc(risk_df, gt_episodes, auc_outcome_names)
    patient_mean      = weighted_mean_auc(patient_auc_table, by="n_pos")
    # Nearest-GT MAE — fair when complications recur (argmax may catch the
    # second occurrence and still be a correct hit). RELEASE excluded — its
    # discharge timing is captured by length_of_stay_mae instead (cleaner
    # length-of-stay regression than a risk-curve-peak MAE).
    peak_mae_table    = time_accuracy_nearest(risk_df, gt_episodes, auc_outcome_names)
    # Length-of-stay MAE — replaces RELEASE peak-MAE.
    los_stats         = length_of_stay_mae(risk_df, gt_episodes)
    print(f"[Eval] Length-of-stay MAE: {los_stats['mae_hours']:.2f}h "
          f"(median {los_stats['median_hours']:.1f}h, p90 {los_stats['p90_hours']:.1f}h, "
          f"n={los_stats['n_patients']}, GT mean {los_stats['gt_mean_hours']:.1f}h, "
          f"pred mean {los_stats['pred_mean_hours']:.1f}h)")

    # Legacy per-window AUC table kept for back-compat and supplementary
    # reporting; no longer the headline.
    auc_table = pooled_episode_auc(risk_df, gt_episodes, outcome_names,
                                    patient_horizons=patient_horizons)
    multi_horizon_table = pooled_auc_across_horizons(
        risk_df, gt_episodes, outcome_names, eval_ds_full,
        horizon_caps_hrs=(48, 168, 336),
    )
    mae_table = time_accuracy(risk_df, gt_first, outcome_names)
    gen_stats = compute_gen_stats(risk_df, patient_horizons=patient_horizons)

    mean_auroc     = float(auc_table["auroc"].mean(skipna=True))
    mean_auprc     = float(auc_table["auprc"].mean(skipna=True))
    mean_mae_hours = float(mae_table["mae_hours"].mean(skipna=True))

    # Summarise per-outcome for the log
    print("[Eval] Per-patient AUC + F1 (new headline framing):")
    for outcome, row in patient_auc_table.iterrows():
        if not np.isnan(row["auroc"]):
            print(f"  {outcome:<45} AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}  "
                  f"maxF1={row['max_f1']:.3f}(τ={row['max_f1_threshold']:.3f})  "
                  f"F1@0.5={row['f1_at_0_5']:.3f}  "
                  f"n_pos={int(row['n_pos'])}  prev={row['prevalence']:.3f}")
    print(f"[Eval] Patient-level mean (support-weighted): AUROC={patient_mean['auroc_weighted']:.3f}  "
          f"AUPRC={patient_mean['auprc_weighted']:.3f}  maxF1={patient_mean['max_f1_weighted']:.3f}  "
          f"F1@0.5={patient_mean['f1_at_0_5_weighted']:.3f}  "
          f"(simple AUROC={patient_mean['auroc_simple']:.3f} / AUPRC={patient_mean['auprc_simple']:.3f} / "
          f"maxF1={patient_mean['max_f1_simple']:.3f} / F1@0.5={patient_mean['f1_at_0_5_simple']:.3f}, "
          f"n_outcomes={patient_mean['n_outcomes_used']})")
    print("[Eval] Per-outcome AUROC (legacy horizon-extended window pooling):")
    for outcome, row in auc_table.iterrows():
        if not np.isnan(row["auroc"]):
            print(f"  {outcome:<45} AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}")
    print(f"[Eval] Generation stats: median_steps={gen_stats.get('gen_median_steps', '-')}, "
          f"median_hours={gen_stats.get('gen_median_hours', '-')}, "
          f"frac_terminal_first24h={gen_stats.get('gen_frac_terminal_first24h', '-')}, "
          f"length_mae_hrs={gen_stats.get('gen_length_mae_hrs', '-')}")
    # Multi-horizon mean — quick read on where the model is good vs collapsed.
    print("[Eval] Multi-horizon mean (across all outcomes with sufficient positives):")
    for cap, sub in multi_horizon_table.groupby("horizon_cap_hrs"):
        m_auroc = float(sub["auroc"].mean(skipna=True))
        m_auprc = float(sub["auprc"].mean(skipna=True))
        print(f"  cap={cap:>3d}h   AUROC={m_auroc:.3f}   AUPRC={m_auprc:.3f}")

    return dict(
        # New headline (patient-level peak-detector).
        patient_auc_table=patient_auc_table,
        patient_auroc_weighted=patient_mean["auroc_weighted"],
        patient_auprc_weighted=patient_mean["auprc_weighted"],
        patient_auroc_simple=patient_mean["auroc_simple"],
        patient_auprc_simple=patient_mean["auprc_simple"],
        # F1 metrics (for direct comparability with F1-reporting EHR literature).
        patient_max_f1_weighted=patient_mean["max_f1_weighted"],
        patient_max_f1_simple=patient_mean["max_f1_simple"],
        patient_f1_at_0_5_weighted=patient_mean["f1_at_0_5_weighted"],
        patient_f1_at_0_5_simple=patient_mean["f1_at_0_5_simple"],
        n_outcomes_used=patient_mean["n_outcomes_used"],
        peak_mae_table=peak_mae_table,
        # Length-of-stay regression (replaces RELEASE peak-MAE).
        length_of_stay_mae_hours=los_stats["mae_hours"],
        length_of_stay_median_hours=los_stats["median_hours"],
        length_of_stay_p90_hours=los_stats["p90_hours"],
        length_of_stay_n_patients=los_stats["n_patients"],
        # Legacy per-window framing (kept for back-compat / supplementary).
        mean_auroc=mean_auroc,
        mean_auprc=mean_auprc,
        mean_mae_hours=mean_mae_hours,
        auc_table=auc_table,
        mae_table=mae_table,
        gen_stats=gen_stats,
        multi_horizon_table=multi_horizon_table,
        # Raw per-step generation output + ground-truth episodes — consumed by
        # bootstrap_evaluate() and by any downstream caller that wants to recompute
        # custom metrics without re-running generate().
        risk_df=risk_df,
        gt_episodes=gt_episodes,
        gt_first=gt_first,
    )


# ===========================================================================
# Bootstrap variance for a trained checkpoint
# ===========================================================================

def _f1_from_scores(scores, labels):
    """
    Purpose: Compute max-F1 (PR-sweep) and F1@0.5 from a (scores, labels) pair.
    Method:  precision_recall_curve to enumerate operating points; max-F1 over
             the swept thresholds; F1@0.5 from the fixed 0.5 cut directly so it
             matches the point-estimate definition used by `per_patient_max_auc`.
             Returns NaN for either when n_pos==0 or n_neg==0.

    Args:
        scores (np.ndarray): per-patient peak P_<outcome> over the resample.
        labels (np.ndarray): {0,1} labels aligned with scores.

    Returns:
        (max_f1, f1_at_0_5) as floats (or NaN if degenerate).
    """
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    prec, rec, _ = precision_recall_curve(labels, scores)
    denom = prec + rec
    f1_arr = np.zeros_like(prec)
    nz = denom > 0
    f1_arr[nz] = 2 * prec[nz] * rec[nz] / denom[nz]
    max_f1 = float(f1_arr.max())
    pred_05 = scores >= 0.5
    tp = int((pred_05 & (labels == 1)).sum())
    fp = int((pred_05 & (labels == 0)).sum())
    fn = int((~pred_05 & (labels == 1)).sum())
    prec_05 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_05  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_05 = 2 * prec_05 * rec_05 / (prec_05 + rec_05) if (prec_05 + rec_05) > 0 else 0.0
    return max_f1, float(f1_05)


def bootstrap_evaluate(model, tokenizer, test_temporal_raw, test_ctx_raw,
                       scaler, checkpoint_dir, B=2000, seed=42):
    """
    Purpose: Patient-level bootstrap CIs for the locked test-set headline.
    Method:  Run `evaluate_on_test_set` ONCE to get the per-step risk_df + GT,
             collapse to per-(patient, outcome) peak scores, then resample
             held-out test patients with replacement (B reps) to produce 95%
             percentile CIs for the support-weighted AUROC / AUPRC / max-F1 /
             F1@0.5 headline, per-outcome AUROC/AUPRC/max-F1/F1@0.5/peak-MAE,
             and length-of-stay MAE. Single model, single generation pass —
             far cheaper than re-seeding the full pipeline. Point estimates
             are unchanged.

    Args:
        model              : Trained InterveneGPT (best available checkpoint).
        tokenizer          : EMRTokenizer matching the training vocab.
        test_temporal_raw  : held-out test split temporal DataFrame.
        test_ctx_raw       : held-out test split context DataFrame.
        scaler             : fitted scaler (joblib-loaded).
        checkpoint_dir (str): for evaluate_on_test_set's DataProcessor path.
        B (int)            : number of bootstrap resamples (default 2000).
        seed (int)         : RNG seed for reproducibility.

    Returns:
        dict: evaluate_on_test_set's output extended with *_ci_lo / *_ci_hi /
        *_boot_mean / *_boot_sd entries and a per_outcome_ci DataFrame (with
        auroc/auprc/max_f1/f1_at_0_5/peak_mae columns). Also prints a grep-
        friendly bootstrap summary block to stdout.
    """
    import time

    res = evaluate_on_test_set(
        model=model, tokenizer=tokenizer,
        test_temporal_raw=test_temporal_raw, test_ctx_raw=test_ctx_raw,
        scaler=scaler, checkpoint_dir=checkpoint_dir,
    )
    risk_df       = res["risk_df"]
    gt_episodes   = res["gt_episodes"]
    outcome_names = [n for n in model.outcome_names if n not in AUC_EXCLUDE]

    # Per-patient max score per outcome — pool the generated (non-input) rows
    # only; mirrors per_patient_max_auc's peak-detector framing.
    gen_df   = risk_df[risk_df["IsInput"] == 0]
    all_pids = list(risk_df["PatientId"].unique())
    N        = len(all_pids)
    p_cols   = [f"P_{n}" for n in outcome_names]

    maxpp = {pid: {c: 0.0 for c in p_cols} for pid in all_pids}
    if len(gen_df):
        g = gen_df.groupby("PatientId")[p_cols].max()
        for pid, row in g.iterrows():
            for c in p_cols:
                maxpp[pid][c] = float(row[c])

    # Per-outcome peak-time per patient (argmax P_outcome over generated rows),
    # used to build a per-outcome time-error array of length N. NaN where the
    # patient has no GT episode of that outcome or generated no rows. Inside
    # each bootstrap resample, NaN entries are filtered, so the peak-MAE CI is
    # over the resample's positive-cohort intersection — same convention as
    # `time_accuracy_nearest()`.
    peak_time = {name: {} for name in outcome_names}
    if len(gen_df):
        idxmax = gen_df.groupby("PatientId")[p_cols].idxmax()
        for name in outcome_names:
            pcol = f"P_{name}"
            ser = idxmax[pcol].dropna().astype(int)
            if len(ser):
                pt = gen_df.loc[ser, ["PatientId", "TimePoint"]].set_index("PatientId")["TimePoint"]
                peak_time[name] = pt.to_dict()

    cols = {}
    time_err = {}
    for name in outcome_names:
        scores = np.array([maxpp[p][f"P_{name}"] for p in all_pids])
        labels = np.array(
            [int(len(gt_episodes.get(p, {}).get(name, [])) > 0) for p in all_pids],
            dtype=np.int64,
        )
        cols[name] = (scores, labels)
        errs = np.full(N, np.nan, dtype=float)
        pt_map = peak_time.get(name, {})
        for i, pid in enumerate(all_pids):
            episodes = gt_episodes.get(pid, {}).get(name, [])
            if not episodes or pid not in pt_map:
                continue
            pt_val = float(pt_map[pid])
            errs[i] = min(abs(pt_val - float(t_gt)) for t_gt in episodes)
        time_err[name] = errs

    min_pos = _min_positives(N)

    per_out = {name: {"auroc": [], "auprc": [], "max_f1": [],
                      "f1_at_0_5": [], "peak_mae": []} for name in cols}
    boot_auroc, boot_auprc = [], []
    boot_maxf1_w, boot_f105_w = [], []
    rng = np.random.RandomState(seed)
    t0  = time.time()
    for _ in range(B):
        idx = rng.randint(0, N, size=N)
        aurocs, auprcs, maxf1s, f105s, weights = [], [], [], [], []
        for nm, (sc, lb) in cols.items():
            s, l = sc[idx], lb[idx]
            n_pos = int(l.sum()); n_neg = len(l) - n_pos
            if n_pos < min_pos or n_neg < min_pos:
                continue
            au = roc_auc_score(l, s)
            ap = average_precision_score(l, s)
            mf1, f105 = _f1_from_scores(s, l)
            aurocs.append(au); auprcs.append(ap)
            maxf1s.append(mf1); f105s.append(f105)
            weights.append(n_pos)
            per_out[nm]["auroc"].append(au)
            per_out[nm]["auprc"].append(ap)
            per_out[nm]["max_f1"].append(mf1)
            per_out[nm]["f1_at_0_5"].append(f105)
            te = time_err[nm][idx]
            valid = te[~np.isnan(te)]
            if valid.size > 0:
                per_out[nm]["peak_mae"].append(float(valid.mean()))
        if weights:
            w = np.array(weights, float); w /= w.sum()
            boot_auroc.append(float((np.array(aurocs) * w).sum()))
            boot_auprc.append(float((np.array(auprcs) * w).sum()))
            boot_maxf1_w.append(float((np.array(maxf1s) * w).sum()))
            boot_f105_w.append(float((np.array(f105s) * w).sum()))
    print(f"[boot] {B} resamples in {time.time()-t0:.1f}s")

    # Length-of-stay bootstrap. Decoder LoS = trajectory-length (last TimePoint)
    # vs first GT RELEASE — matches length_of_stay_mae() upstream. RELEASE-only
    # cohort; patients without GT release are excluded.
    los_pairs = []
    for pid, sub in risk_df.groupby("PatientId"):
        gt_releases = gt_episodes.get(pid, {}).get("RELEASE_EVENT", [])
        if not gt_releases:
            continue
        pred_los = float(sub["TimePoint"].max())
        gt_los   = float(min(gt_releases))
        los_pairs.append(abs(pred_los - gt_los))
    los_arr = np.asarray(los_pairs)

    boot_los = []
    rng2 = np.random.RandomState(seed + 1)
    if los_arr.size:
        nL = los_arr.size
        for _ in range(B):
            boot_los.append(los_arr[rng2.randint(0, nL, size=nL)].mean())

    def _ci(arr):
        a = np.asarray(arr)
        return np.percentile(a, 2.5), np.percentile(a, 97.5), a.mean(), a.std()

    point_auroc = res["patient_auroc_weighted"]
    point_auprc = res["patient_auprc_weighted"]
    point_maxf1 = res["patient_max_f1_weighted"]
    point_f105  = res["patient_f1_at_0_5_weighted"]
    print(f"\n=== BOOTSTRAP 95pct CI (patient resample, B={B}) ===")
    print(f"[boot] point estimate: AUROC_w={point_auroc:.4f}  "
          f"AUPRC_w={point_auprc:.4f}  "
          f"maxF1_w={point_maxf1:.4f}  F1@0.5_w={point_f105:.4f}  "
          f"N_test={N}")

    out = dict(res)
    for label, point, arr in [
        ("patient_auroc_weighted",     point_auroc, boot_auroc),
        ("patient_auprc_weighted",     point_auprc, boot_auprc),
        ("patient_max_f1_weighted",    point_maxf1, boot_maxf1_w),
        ("patient_f1_at_0_5_weighted", point_f105,  boot_f105_w),
    ]:
        if not arr:
            print(f"{label}: (insufficient successful resamples)")
            continue
        lo, hi, mean, sd = _ci(arr)
        out[f"{label}_ci_lo"]     = float(lo)
        out[f"{label}_ci_hi"]     = float(hi)
        out[f"{label}_boot_mean"] = float(mean)
        out[f"{label}_boot_sd"]   = float(sd)
        print(f"{label}: point={point:.4f}  boot_mean={mean:.4f}  "
              f"95%CI=[{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")

    if boot_los:
        lo, hi, mean, sd = _ci(boot_los)
        out["length_of_stay_mae_hours_ci_lo"]     = float(lo)
        out["length_of_stay_mae_hours_ci_hi"]     = float(hi)
        out["length_of_stay_mae_hours_boot_mean"] = float(mean)
        out["length_of_stay_mae_hours_boot_sd"]   = float(sd)
        print(f"length_of_stay_mae_hours (RELEASE-only, n={los_arr.size}): "
              f"point={res['length_of_stay_mae_hours']:.4f}  "
              f"boot_mean={mean:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")

    # Per-outcome CI table. Reports AUROC/AUPRC/max-F1/F1@0.5 (rank+threshold
    # metrics, resampled jointly per outcome) and the per-outcome peak-MAE
    # (resampled over the positive-cohort intersection of each resample).
    print("\n--- per-outcome 95% CI ---")
    print(f"{'outcome':<32}{'AUROC':<22}{'AUPRC':<22}"
          f"{'maxF1':<22}{'F1@0.5':<22}{'peak-MAE (h)'}")
    per_out_rows = []
    for name in cols:
        ar = per_out[name]["auroc"]
        pr = per_out[name]["auprc"]
        mf = per_out[name]["max_f1"]
        ff = per_out[name]["f1_at_0_5"]
        pm = per_out[name]["peak_mae"]
        if not ar:
            print(f"{name:<32}(insufficient positives in resamples)")
            per_out_rows.append({"outcome": name,
                                 "auroc_mean": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
                                 "auprc_mean": np.nan, "auprc_lo": np.nan, "auprc_hi": np.nan,
                                 "max_f1_mean": np.nan, "max_f1_lo": np.nan, "max_f1_hi": np.nan,
                                 "f1_at_0_5_mean": np.nan, "f1_at_0_5_lo": np.nan, "f1_at_0_5_hi": np.nan,
                                 "peak_mae_mean": np.nan, "peak_mae_lo": np.nan, "peak_mae_hi": np.nan})
            continue
        alo, ahi, am, _ = _ci(ar)
        plo, phi, pmean, _ = _ci(pr)
        mlo, mhi, mm, _ = _ci(mf)
        flo, fhi, fm, _ = _ci(ff)
        if pm:
            plm_lo, plm_hi, plm_mean, _ = _ci(pm)
        else:
            plm_lo, plm_hi, plm_mean = float("nan"), float("nan"), float("nan")
        per_out_rows.append({"outcome": name,
                             "auroc_mean": float(am),  "auroc_lo": float(alo),  "auroc_hi": float(ahi),
                             "auprc_mean": float(pmean), "auprc_lo": float(plo), "auprc_hi": float(phi),
                             "max_f1_mean": float(mm),  "max_f1_lo": float(mlo),  "max_f1_hi": float(mhi),
                             "f1_at_0_5_mean": float(fm), "f1_at_0_5_lo": float(flo), "f1_at_0_5_hi": float(fhi),
                             "peak_mae_mean": float(plm_mean), "peak_mae_lo": float(plm_lo), "peak_mae_hi": float(plm_hi)})
        pm_str = (f"{plm_mean:.2f} [{plm_lo:.2f},{plm_hi:.2f}]"
                  if not np.isnan(plm_mean) else "—")
        print(f"{name:<32}"
              f"{am:.3f} [{alo:.3f},{ahi:.3f}]  "
              f"{pmean:.3f} [{plo:.3f},{phi:.3f}]  "
              f"{mm:.3f} [{mlo:.3f},{mhi:.3f}]  "
              f"{fm:.3f} [{flo:.3f},{fhi:.3f}]  "
              f"{pm_str}")
    out["per_outcome_ci"] = pd.DataFrame(per_out_rows).set_index("outcome")
    print("\n[boot] done.")
    return out
