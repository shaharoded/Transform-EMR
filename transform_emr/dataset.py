import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *


class EMRDataset(Dataset):
    def __init__(self, df, patient_context_df, scaler=None, max_input_days=None):
        """
        df: original DataFrame with columns ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        patient_context_df: DataFrame with columns ['PatientID'] + context_columns.
        max_input_days: For test dataset - will truncate the input temporal records after |max_input_days| days.

        This class performs data cleaning, as well as prperation of data for input as train of for inference as test.

        Attr:
            self.scaler (sklearn.preprocessing.StandardScaler): Fitted scaler used to normalize the patient context vectors.
            self.context_df (pd.DataFrame): Patient-level context features (indexed by PatientID), scaled to zero mean and unit variance.
            self.tokens_df (pd.DataFrame): Long-format temporal event dataframe with per-token attributes and timing features.
            self.patient_ids (np.ndarray): Array of unique PatientIDs present in the dataset.
            self.patient_groups (Dict[str, pd.DataFrame]): Mapping from PatientID to their corresponding token DataFrame.

            self.concept2id (Dict[str, int]): Vocabulary mapping for unique concept names.
            self.value2id (Dict[str, int]): Vocabulary mapping for concept+value tokens.
            self.token2id (Dict[str, int]): Vocabulary mapping for concept+value+position tokens (used as model input/target).
            self.id2concept (Dict[int, str]): Reverse mapping for concept2id.
            self.id2value (Dict[int, str]): Reverse mapping for value2id.
            self.id2token (Dict[int, str]): Reverse mapping for position2id.

            self.token_weights (torch.Tensor): A token importance weights matrix for loss function
        """
        # Input validation
        df, patient_context_df = self._validate_and_align_inputs(df, patient_context_df)
        df = self._truncate_after_terminal_event(df)
        
        # Set context
        self.context_df = patient_context_df.set_index("PatientID").astype("float32")
        
        # Fit scaler and transform context features
        self.scaler = scaler if scaler else StandardScaler()
        self.context_df.loc[:, :] = self.scaler.fit_transform(self.context_df.values)

        # Normalize time (based on admission)
        df["IsAdmission"] = df["ConceptName"] == ADMISSION_TOKEN
        df["VisitCounter"] = df.groupby("PatientID")["IsAdmission"].cumsum()
        df["VisitID"] = df["PatientID"].astype(str) + "_" + df["VisitCounter"].astype(str)
        df["VisitStart"] = df.groupby("VisitID")["StartDateTime"].transform('min')
        df["RelStartTime"] = (df["StartDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        df["RelEndTime"] = (df["EndDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours

        if max_input_days:
            df = self._cut_after_k_days(df, max_input_days)

        # Expand tokens
        self.tokens_df = self._expand_tokens(df)

        # Create token vocabulary
        raw_concepts = sorted(self.tokens_df['RawConcept'].unique())
        concepts = sorted(self.tokens_df['Concept'].unique())
        values = sorted(self.tokens_df['ValueToken'].unique())
        positions = sorted(self.tokens_df['PositionToken'].unique())  # already being used

        special_tokens = ["[PAD]", "[CTX]", "[MASK]"]
        positions = [tok for tok in positions if tok not in special_tokens]  # don't double add
        positions = special_tokens + positions

        # Build vocab mappings
        self.rawconcept2id = {tok: i for i, tok in enumerate(raw_concepts)}
        self.concept2id = {tok: i for i, tok in enumerate(concepts)}
        self.value2id = {tok: i for i, tok in enumerate(values)}
        self.token2id = {tok: i for i, tok in enumerate(positions)}

        # Reverse lookups
        self.id2rawconcept = {v: k for k, v in self.rawconcept2id.items()}
        self.id2concept = {v: k for k, v in self.concept2id.items()}
        self.id2value = {v: k for k, v in self.value2id.items()}
        self.id2token = {v: k for k, v in self.token2id.items()}

        # Map onto df
        self.tokens_df['RawConceptID'] = self.tokens_df['RawConcept'].map(self.rawconcept2id)
        self.tokens_df['ConceptID'] = self.tokens_df['Concept'].map(self.concept2id)
        self.tokens_df['ValueID'] = self.tokens_df['ValueToken'].map(self.value2id)
        self.tokens_df['PositionID'] = self.tokens_df['PositionToken'].map(self.token2id)

        # Sort & compute time deltas
        self.tokens_df = self.tokens_df.sort_values(['PatientID', 'TimePoint'])
        self.tokens_df['TimeDelta'] = self.tokens_df.groupby('PatientID')['TimePoint'].diff().fillna(0)

        # Token Weights (for loss function)
        self.token_weights = torch.ones(len(self.token2id))

        # Boost OUTCOMES
        for outcome in OUTCOMES:
            tok_id = self.token2id.get(outcome)
            if tok_id is not None:
                self.token_weights[tok_id] = 5.0

        # Boost TERMINAL outcomes heavily
        for term in TERMINAL_OUTCOMES:
            tok_id = self.token2id.get(term)
            if tok_id is not None:
                self.token_weights[tok_id] = 15.0

        # Boost MEAL tokens moderately
        for tok in self.token2id:
            if "MEAL" in tok:
                self.token_weights[self.token2id[tok]] = 3.0

        # Boost interval tokens (START/END)
        for tok in self.token2id:
            if tok.endswith("_START") or tok.endswith("_END"):
                self.token_weights[self.token2id[tok]] = 2.5

        # Suppress special/control tokens
        for ignore_tok in special_tokens + [ADMISSION_TOKEN]:
            tok_id = self.token2id.get(ignore_tok)
            if tok_id is not None:
                self.token_weights[tok_id] = 0.0

        # Merge patient context
        self.patient_ids = self.tokens_df['PatientID'].unique()
        self.patient_groups = {pid: self.tokens_df[self.tokens_df['PatientID'] == pid] for pid in self.patient_ids}

    def _validate_and_align_inputs(self, df, patient_context_df):
        """
        Validates required columns, datetime types, and aligns PatientIDs between
        temporal (df) and context (patient_context_df) data. Will also sort the temporal data.

        Returns:
            Tuple of (cleaned_df, cleaned_patient_context_df)
        """

        # 1. Required columns check
        required_columns = ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column in temporal data: {col}")
        if 'PatientID' not in patient_context_df.columns:
            raise ValueError("Missing 'PatientID' column in context data")

        # 2. Check datetime dtypes
        if not pd.api.types.is_datetime64_any_dtype(df['StartDateTime']):
            raise TypeError("StartDateTime column must be of datetime64[ns] dtype.")
        if not pd.api.types.is_datetime64_any_dtype(df['EndDateTime']):
            raise TypeError("EndDateTime column must be of datetime64[ns] dtype.")

        # 3. Handle duplicate PatientIDs in context
        dupe_counts = patient_context_df['PatientID'].value_counts()
        duplicates = dupe_counts[dupe_counts > 1]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} PatientIDs with duplicate rows in context_df. Aggregating by max value...")
            patient_context_df = patient_context_df.groupby('PatientID').max().reset_index()

        # 4. Align temporal and context data
        temporal_ids = set(df['PatientID'])
        context_ids = set(patient_context_df['PatientID'])
        shared_ids = temporal_ids & context_ids

        if len(shared_ids) < len(temporal_ids) or len(shared_ids) < len(context_ids):
            print(f"Dropping unmatched PatientIDs:")
            print(f"   - {len(temporal_ids - shared_ids)} from temporal data")
            print(f"   - {len(context_ids - shared_ids)} from context data")

            df = df[df['PatientID'].isin(shared_ids)].copy()
            patient_context_df = patient_context_df[patient_context_df['PatientID'].isin(shared_ids)].copy()

        # 5. Final integrity checks
        assert patient_context_df['PatientID'].is_unique, "PatientID must be unique in context_df after aggregation"
        assert set(df['PatientID']) == set(patient_context_df['PatientID']), "Mismatched PatientIDs after filtering"

        return df.sort_values(["PatientID", "StartDateTime"]).copy(), patient_context_df
    
    def _truncate_after_terminal_event(self, df):
        """
        For each patient:
        - Drop RELEASE_TOKEN if a DEATH_TOKEN occurs within 30 days after it.
        - Then truncate any records after the first terminal event.
        """

        def process_group(group):
            group = group.sort_values("StartDateTime").copy()

            # Handle RELEASE vs DEATH conflicts
            release_rows = group[group["ConceptName"] == RELEASE_TOKEN]
            death_rows = group[group["ConceptName"] == DEATH_TOKEN]

            if not release_rows.empty and not death_rows.empty:
                release_time = release_rows.iloc[0]["StartDateTime"]
                death_time = death_rows.iloc[0]["StartDateTime"]
                # If death is within 30 days after release → drop release
                if pd.Timedelta(0) <= (death_time - release_time) <= pd.Timedelta(days=30):
                    group = group[group["ConceptName"] != RELEASE_TOKEN]

            # Then truncate after first terminal event
            terminal_idx = group[group["ConceptName"].isin(TERMINAL_OUTCOMES)].index
            if not terminal_idx.empty:
                first_terminal_time = group.loc[terminal_idx[0], "StartDateTime"]
                group = group[group["StartDateTime"] <= first_terminal_time]

            return group

        df = df.groupby("PatientID", group_keys=False).apply(process_group).reset_index(drop=True)
        return df
    
    def _cut_after_k_days(self, df, k_days):
        """
        Trims patient timelines to only include events within the first `k` days from admission.
        Drops patients whose entire stay is <= k+1 days (nothing to predict beyond that).
        """
        k_hours = k_days * 24

        # Keep only visits with at least k+1 minutes
        visit_durations = df.groupby("VisitID")["RelStartTime"].max()
        eligible_visits = visit_durations[visit_durations > k_hours].index
        df = df[df["VisitID"].isin(eligible_visits)].copy()

        # Cut timeline to first k minutes of visit
        df = df[df["RelStartTime"] <= k_hours].copy()

        # Drop visits with 1 or fewer records
        visit_counts = df.groupby("VisitID").size()
        df = df[df["VisitID"].isin(visit_counts[visit_counts > 1].index)]

        return df
    
    def _expand_tokens(self, df, min_state_duration_sec=1):
        """
        Expands events into tokens with timepoints.

        - Splits state events into START and END tokens.
        - Keeps instantaneous events as single tokens.
        
        Returns:
            DataFrame with ['PatientID', 'RawConcept', 'Concept', 'ValueToken', 'PositionToken', 'TimePoint'].
        """
        rows = []
        for _, row in df.iterrows():
            duration_sec = (row['EndDateTime'] - row['StartDateTime']).total_seconds()
            is_state = duration_sec > min_state_duration_sec

            base_token = f"{row['ConceptName']}_{row['Value']}" if row['Value'] not in ("True", "TRUE") else row['ConceptName']
            concept = row['ConceptName']
            value = base_token
            if concept.endswith(('_STATE', '_TREND')):
                raw_concept = concept.rsplit('_', 1)[0]
            else:
                raw_concept = concept
            pos_tokens = []

            if is_state:
                pos_tokens = ["START", "END"]
                time_points = [row['RelStartTime'], row['RelEndTime']]
            else:
                pos_tokens = [""]
                time_points = [row['RelStartTime']]

            for pos, tp in zip(pos_tokens, time_points):
                full_token = f"{base_token}_{pos}" if pos else base_token
                rows.append({
                    'PatientID': row['PatientID'],
                    'RawConcept': raw_concept,
                    'Concept': concept,
                    'ValueToken': value,
                    'PositionToken': full_token,
                    'TimePoint': tp
                })

        return pd.DataFrame(rows)

    def __len__(self):
        """
        Returns the number of patients in the dataset
        """
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """
        Returns the subst of records for 1 patient.
        """
        pid = self.patient_ids[idx]
        df = self.patient_groups[pid]

        return {
            "raw_concept_ids": torch.tensor(df["RawConceptID"].values, dtype=torch.long),
            "concept_ids": torch.tensor(df["ConceptID"].values, dtype=torch.long),
            "value_ids": torch.tensor(df["ValueID"].values, dtype=torch.long),
            "position_ids": torch.tensor(df["PositionID"].values, dtype=torch.long),
            "delta_ts": torch.tensor(df["TimeDelta"].values, dtype=torch.float32),
            "abs_ts": torch.tensor(df["TimePoint"].values, dtype=torch.float32),
            "context_vec": torch.tensor(self.context_df.loc[pid].values, dtype=torch.float32),
            "targets": torch.tensor(df["PositionID"].values, dtype=torch.long),  # next-token target
        }

def collate_emr(batch, pad_token_id=0):
    """
    Collates a batch of patient EMR sequences into padded tensors.

    Each sequence contains:
        - Concept ID
        - Value ID
        - Position ID (used for prediction)
        - Delta time (Δt)
        - Absolute time (since admission)
        - Patient context vector (no padding)

    Returns:
        Dictionary of padded tensors: [B, T_max] + context_vec [B, C]
    """
    batch_size = len(batch)
    max_len = max(len(x['position_ids']) for x in batch)

    def pad_tensor(seq, pad_val=0, dtype=torch.long):
        out = torch.full((batch_size, max_len), pad_val, dtype=dtype)
        for i, s in enumerate(seq):
            out[i, :len(s)] = s
        return out

    raw_concept_ids  = pad_tensor([x['raw_concept_ids'] for x in batch], pad_val=pad_token_id)
    concept_ids      = pad_tensor([x['concept_ids'] for x in batch], pad_val=pad_token_id)
    value_ids        = pad_tensor([x['value_ids'] for x in batch], pad_val=pad_token_id)
    position_ids     = pad_tensor([x['position_ids'] for x in batch], pad_val=pad_token_id)
    delta_ts         = pad_tensor([x['delta_ts'] for x in batch], pad_val=0.0, dtype=torch.float32)
    abs_ts           = pad_tensor([x['abs_ts'] for x in batch], pad_val=0.0, dtype=torch.float32)

    context_vecs = torch.stack([x['context_vec'] for x in batch])

    return {
        'raw_concept_ids': raw_concept_ids,
        'concept_ids': concept_ids,
        'value_ids': value_ids,
        'position_ids': position_ids,  # targets
        'delta_ts': delta_ts,
        'abs_ts': abs_ts,
        'context_vec': context_vecs,
        'targets': position_ids.clone()
    }