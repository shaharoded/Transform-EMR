import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *
from transform_emr.config.model_config import CHECKPOINT_PATH


class DataProcessor:
    """
    Handles the dataprocess needed to build the tokenizer / train / val / test.
    use max_input_days to trim a test dataset before using it for prediction.

    Expected columns for temporal_df: ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
    Expected columns for context_df: ['PatientID'] + context columns.

    Attributes:
    df (pd.DataFrame): Transformed long-format event dataframe after all processing.
    context_df (pd.DataFrame): Patient context dataframe with PatientID as index.
    scaler (StandardScaler): Scaler fitted to context_df and optionally saved to disk.

    """
    def __init__(self, df, context_df, max_input_days=None, scaler=None):
        df['StartDateTime'] = pd.to_datetime(df['StartTime'], utc=True, errors='raise')
        df['StartDateTime'] = df['StartDateTime'].dt.tz_convert(None)
        df['EndDateTime'] = pd.to_datetime(df['EndTime'], utc=True, errors='raise')
        df['EndDateTime'] = df['EndDateTime'].dt.tz_convert(None)
        df.drop(columns=["StartTime", "EndTime"], inplace=True)

        self.df = df.copy()
        self.context_df = context_df.copy()
        self.max_input_days = max_input_days
        self.scaler = scaler


    def run(self):
        self._validate_and_align_inputs()
        self._truncate_after_terminal_event()
        self._normalize_time()
        if self.max_input_days:
            self._cut_after_k_days()
        self._expand_tokens()
        self.context_df = self.context_df.set_index("PatientID").astype("float32")
        self._fit_scaler()
        return self.df, self.context_df

    def _fit_scaler(self):
        """
        Fit and / or use a standard scaler on the context dataframe. 
        Will save the scaler in the checkpoints.
        """        
        if self.scaler is None:
            scaler = StandardScaler()
            self.context_df.loc[:, :] = scaler.fit_transform(self.context_df.values)
            dump(scaler, os.path.join(CHECKPOINT_PATH, 'scaler.pkl'))
        else:
            self.context_df.loc[:, :] = self.scaler.transform(self.context_df.values)

    def _validate_and_align_inputs(self):
        """
        Validates required columns, datetime types, and aligns PatientIDs between
        temporal (df) and context (patient_context_df) data. Will also sort the temporal data.

        Returns:
            Tuple of (cleaned_df, cleaned_patient_context_df)
        """
        # 1. Required columns check
        required_columns = ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column in temporal data: {col}")
        if 'PatientID' not in self.context_df.columns:
            raise ValueError("Missing 'PatientID' column in context data")

        # 2. Check datetime dtypes
        if not pd.api.types.is_datetime64_any_dtype(self.df['StartDateTime']):
            raise TypeError("StartDateTime column must be of datetime64[ns] dtype.")
        if not pd.api.types.is_datetime64_any_dtype(self.df['EndDateTime']):
            raise TypeError("EndDateTime column must be of datetime64[ns] dtype.")

        # 3. Handle duplicate PatientIDs in context
        dupe_counts = self.context_df['PatientID'].value_counts()
        duplicates = dupe_counts[dupe_counts > 1]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} PatientIDs with duplicate rows in context_df. Aggregating by max value...")
            self.context_df = self.context_df.groupby('PatientID').max().reset_index()

        # 4. Align temporal and context data
        temporal_ids = set(self.df['PatientID'])
        context_ids = set(self.context_df['PatientID'])
        shared_ids = temporal_ids & context_ids

        if len(shared_ids) < len(temporal_ids) or len(shared_ids) < len(context_ids):
            print(f"Dropping unmatched PatientIDs:")
            print(f"   - {len(temporal_ids - shared_ids)} from temporal data")
            print(f"   - {len(context_ids - shared_ids)} from context data")

            self.df = self.df[self.df['PatientID'].isin(shared_ids)].copy()
            self.context_df = self.context_df[self.context_df['PatientID'].isin(shared_ids)].copy()

        # 5. Final integrity checks
        assert self.context_df['PatientID'].is_unique, "PatientID must be unique in context_df after aggregation"
        assert set(self.df['PatientID']) == set(self.context_df['PatientID']), "Mismatched PatientIDs after filtering"


    def _truncate_after_terminal_event(self):
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

        self.df = self.df.groupby("PatientID", group_keys=False).apply(process_group).reset_index(drop=True)


    def _cut_after_k_days(self):
        """
        Trims patient timelines to only include events within the first `k` days from admission.
        Drops patients whose entire stay is <= k+1 days (nothing to predict beyond that).
        """
        df = self.df
        k_days = self.max_input_days
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

        self.df = df
    

    def _normalize_time(self):
        df = self.df.copy()
        df["IsAdmission"] = df["ConceptName"] == ADMISSION_TOKEN
        df["VisitCounter"] = df.groupby("PatientID")["IsAdmission"].cumsum()
        df["VisitID"] = df["PatientID"].astype(str) + "_" + df["VisitCounter"].astype(str)
        df["VisitStart"] = df.groupby("VisitID")["StartDateTime"].transform('min')
        df["RelStartTime"] = (df["StartDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        df["RelEndTime"] = (df["EndDateTime"] - df["VisitStart"]).dt.total_seconds() / 3600.0 # In hours
        self.df = df


    def _expand_tokens(self, min_state_duration_sec=1):
        """
        Expands events into tokens with timepoints.

        - Splits state events into START and END tokens.
        - Keeps instantaneous events as single tokens.
        
        Returns:
            DataFrame with ['PatientID', 'RawConcept', 'Concept', 'ValueToken', 'PositionToken', 'TimePoint'].
        """
        df = self.df
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

        self.df = pd.DataFrame(rows)


class EMRTokenizer:
    """
    A custom tokenizer objest to match this model's requirement.
    build this object with your full training data to ensure it builds properly.

    Attributes:
        token2id (Dict[str, int]): Full vocabulary mapping ("GLUCOSE_STATE_HIGH_START").
        id2token (Dict[int, str]): Reverse mapping for decoding.
        rawconcept2id (Dict[str, int]): Vocabulary mapping for raw concepts only ("GLUCOSE").
        concept2id (Dict[str, int]): Vocabulary mapping for concepts only ("GLUCOSE_STATE"/ "GLUCOSE_TREND")..
        value2id (Dict[str, int]): Vocabulary mapping for concepts+values ("GLUCOSE_STATE_HIGH")
        special_tokens (List[str]): Special tokens like ["MASK"].
        token_weights (torch.Tensor): Weights used in loss to emphasize important tokens.
        important_token_ids (torch.Tensor): Token IDs with weight > 1.0.
        pad_token_id (int): ID for padding token.
        mask_token_id (int): ID for mask token.
        ctx_token_id (int): ID for context token.
    """
    def __init__(self, token2id, rawconcept2id, concept2id, value2id, special_tokens, 
                 token_weights, important_token_ids):
        self.token2id = token2id
        self.id2token = {i: tok for tok, i in token2id.items()}
        self.rawconcept2id = rawconcept2id
        self.concept2id = concept2id
        self.value2id = value2id
        self.special_tokens = special_tokens
        self.token_weights = token_weights
        self.important_token_ids = important_token_ids
        self.mask_token_id = token2id["[MASK]"]
        self.pad_token_id = token2id["[PAD]"]
        self.ctx_token_id = token2id["[CTX]"]


    @classmethod
    def from_processed_df(cls, df, special_tokens=["[PAD]", "[MASK]", "[CTX]"]):
        raw_concepts = sorted(df['RawConcept'].unique())
        concepts = sorted(df['Concept'].unique())
        values = sorted(df['ValueToken'].unique())
        positions = sorted(df['PositionToken'].unique())

        positions = [tok for tok in positions if tok not in special_tokens]
        positions = special_tokens + positions

        token2id = {tok: i for i, tok in enumerate(positions)}
        rawconcept2id = {tok: i for i, tok in enumerate(raw_concepts)}
        concept2id = {tok: i for i, tok in enumerate(concepts)}
        value2id = {tok: i for i, tok in enumerate(values)}

        # Initialize weights
        token_weights = torch.ones(len(token2id))

        for outcome in OUTCOMES:
            tok_id = token2id.get(outcome)
            if tok_id is not None:
                token_weights[tok_id] = 5.0
        for term in TERMINAL_OUTCOMES:
            tok_id = token2id.get(term)
            if tok_id is not None:
                token_weights[tok_id] = 15.0
        for tok in token2id:
            if "MEAL" in tok:
                token_weights[token2id[tok]] = 3.0
            if tok.endswith("_START") or tok.endswith("_END"):
                token_weights[token2id[tok]] = 1.5
        for ignore_tok in special_tokens + [ADMISSION_TOKEN]:
            tok_id = token2id.get(ignore_tok)
            if tok_id is not None:
                token_weights[tok_id] = 0.0
        
        important_token_ids = (token_weights > 1.0).nonzero(as_tuple=True)[0]

        return cls(token2id, rawconcept2id, concept2id, value2id, special_tokens, token_weights, important_token_ids)
    
    def save(self, path=os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        torch.save({
            'token2id': self.token2id,
            'rawconcept2id': self.rawconcept2id,
            'concept2id': self.concept2id,
            'value2id': self.value2id,
            'special_tokens': self.special_tokens,
            'token_weights': self.token_weights,
            'important_token_ids': self.important_token_ids 
        }, path)

    @classmethod
    def load(cls, path=os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        obj = torch.load(path)
        return cls(
            token2id=obj['token2id'],
            rawconcept2id=obj['rawconcept2id'],
            concept2id=obj['concept2id'],
            value2id=obj['value2id'],
            special_tokens=obj['special_tokens'],
            token_weights=obj['token_weights'],
            important_token_ids=obj['important_token_ids']
        )
    

class EMRDataset(Dataset):
    def __init__(self, processed_df, context_df, tokenizer):
        """
        processed_df: processed DataFrame after running DataProcessor.run() on the original temporal df.
        context_df: Also processed by DataProcessor.run().

        This class performs data cleaning, as well as prperation of data for input as train of for inference as test.

        Attr:
            self.tokenizer (EMRTokenizer): A tokenizer object capable of encoding and decoding all temporal tokens (and subtokens as required)
            self.context_df (pd.DataFrame): Patient-level context features (indexed by PatientID), scaled to zero mean and unit variance.
            self.tokens_df (pd.DataFrame): Long-format temporal event dataframe with per-token attributes and timing features.
            self.patient_ids (np.ndarray): Array of unique PatientIDs present in the dataset.
            self.patient_groups (Dict[str, pd.DataFrame]): Mapping from PatientID to their corresponding token DataFrame.
        """
        self.tokenizer = tokenizer
        self.tokens_df = processed_df
        self.context_df = context_df

        # Map to token IDs using the tokenizer
        self.tokens_df['RawConceptID'] = self.tokens_df['RawConcept'].map(self.tokenizer.rawconcept2id)
        self.tokens_df['ConceptID'] = self.tokens_df['Concept'].map(self.tokenizer.concept2id)
        self.tokens_df['ValueID'] = self.tokens_df['ValueToken'].map(self.tokenizer.value2id)
        self.tokens_df['PositionID'] = self.tokens_df['PositionToken'].map(self.tokenizer.token2id).fillna(self.tokenizer.mask_token_id).astype(int)

        self.tokens_df = self.tokens_df.sort_values(['PatientID', 'TimePoint'])
        self.tokens_df['TimeDelta'] = self.tokens_df.groupby('PatientID')['TimePoint'].diff().fillna(0)

        self.patient_ids = self.tokens_df['PatientID'].unique()
        self.patient_groups = {pid: self.tokens_df[self.tokens_df['PatientID'] == pid] for pid in self.patient_ids}


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