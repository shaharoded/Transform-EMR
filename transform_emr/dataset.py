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
            self.scaler (sklearn.preprocessing.StandardScaler): Used to scale the ctx vector. Can be inputed externally for validation/testing.
            self.context_df (pd.DataFrame): A non-temporal dataset for global context on the patient.
            self.tokens_df (pd.DataFrame): The temporal dataframe with the longtitudinal records of all the patients.
            self.patient_ids (pd.Series): Unique patient IDs in the data
            self.patient_groups (Dict): {pid: DataFrame} -> A sub-DataFrame containing the records of 1 patient as value
            self.token2id (Dict): token -> ID mapper.
        """
        # Input validation
        df, patient_context_df = self._validate_and_align_inputs(df, patient_context_df)
        df = self._truncate_after_terminal_event(df)
        
        # Set context
        self.context_df = patient_context_df.set_index("PatientID").astype("float32")
        
        # Fit scaler and transform context features
        self.scaler = scaler if scaler else StandardScaler()
        self.context_df.loc[:, :] = self.scaler.fit_transform(self.context_df.values)

        # Normalize time
        df['VisitStart'] = df.groupby('PatientID')['StartDateTime'].transform('min')
        df['RelStartTime'] = (df['StartDateTime'] - df['VisitStart']).dt.total_seconds() / 86400
        df['RelEndTime'] = (df['EndDateTime'] - df['VisitStart']).dt.total_seconds() / 86400

        if max_input_days:
            df = self._cut_after_k_days(df, max_input_days)

        # Expand tokens
        self.tokens_df = self._expand_tokens(df)

        # Create token vocabulary
        special_tokens = ["[PAD]", "[CTX]", "[MASK]"]          # 0, 1, 2
        unique_tokens = self.tokens_df['EventToken'].unique()
        unique_tokens = sorted(set(unique_tokens) - set(special_tokens)) # remove any accidental clashes
        token2id = {tok_id: idx for idx, tok_id in enumerate(special_tokens)}
        token2id.update({tok: i + len(special_tokens) for i, tok in enumerate(unique_tokens)})
        self.token2id = token2id
        self.tokens_df['TokenID'] = self.tokens_df['EventToken'].map(self.token2id)

        # Sort & compute time deltas
        self.tokens_df = self.tokens_df.sort_values(['PatientID', 'TimePoint'])
        self.tokens_df['TimeDelta'] = self.tokens_df.groupby('PatientID')['TimePoint'].diff().fillna(0)

        # Merge patient context
        self.patient_ids = self.tokens_df['PatientID'].unique()
        self.patient_groups = {pid: self.tokens_df[self.tokens_df['PatientID'] == pid] for pid in self.patient_ids}

    def _validate_and_align_inputs(self, df, patient_context_df):
        """
        Validates required columns, datetime types, and aligns PatientIDs between
        temporal (df) and context (patient_context_df) data.

        Returns:
            Tuple of (cleaned_df, cleaned_patient_context_df)
        """

        # 1. Required columns check
        required_columns = ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

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

        return df, patient_context_df
    
    def _truncate_after_terminal_event(self, df):
        """
        For each patient, drop any records that occur after the first terminal event.
        """
        def truncate_group(group):
            group = group.sort_values("StartDateTime")
            terminal_idx = group[group["ConceptName"].isin(TERMINAL_OUTCOMES)].index
            if not terminal_idx.empty:
                first_terminal_time = group.loc[terminal_idx[0], "StartDateTime"]
                return group[group["StartDateTime"] <= first_terminal_time]
            return group

        df = df.groupby("PatientID", group_keys=False).apply(
            lambda group: truncate_group(group)
        )
        return df
    
    def _cut_after_k_days(self, df, k):
        """
        Trims patient timelines to only include events within the first `k` days from admission.
        Drops patients whose entire stay is <= k days (nothing to predict beyond that).
        """
        # Keep events where start is within k days
        df = df[df['RelStartTime'] <= k].copy()

        # Drop patients where we cut the entire timeline (no prediction to do)
        full_counts = df.groupby('PatientID').size()
        valid_ids = full_counts[full_counts > 1].index  # Keep patients with >1 event
        df = df[df['PatientID'].isin(valid_ids)].copy()

        return df
    
    def _expand_tokens(self, df, min_state_duration_sec=1):
        rows = []
        for _, row in df.iterrows():
            duration_sec = (row['EndDateTime'] - row['StartDateTime']).total_seconds()
            is_state = duration_sec > min_state_duration_sec # every event is 1 sec, or more is state / trend

            if is_state:
                rows.append({
                    'PatientID': row['PatientID'],
                    'EventToken': f"{row['ConceptName']}_{row['Value']}_START",
                    'TimePoint': row['RelStartTime']
                })
                rows.append({
                    'PatientID': row['PatientID'],
                    'EventToken': f"{row['ConceptName']}_{row['Value']}_END",
                    'TimePoint': row['RelEndTime']
                })
            else:
                token = row['ConceptName'] if row['Value'] in ("True", "TRUE") else f"{row['ConceptName']}_{row['Value']}"
                rows.append({
                    'PatientID': row['PatientID'],
                    'EventToken': token,
                    'TimePoint': row['RelStartTime']
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

        token_ids = torch.tensor(df['TokenID'].values, dtype=torch.long)
        time_deltas = torch.tensor(df['TimeDelta'].values, dtype=torch.float32)
        context_vector = torch.tensor(self.context_df.loc[pid].values, dtype=torch.float32)

        return {
            'token_ids': token_ids,
            'time_deltas': time_deltas,
            'context_vec': context_vector,
            'targets': token_ids.clone() # For transformer only
        }

def collate_emr(batch, pad_token_id=0):
    """
    Custom collate function for batching EMR sequences of varying lengths.

    This function is designed for use with a DataLoader and the EMRDataset.
    Each patient has a different number of events, so we must:
    - Pad all token and time sequences in the batch to the same length.
    - Ensure the model can process the batch as a [B, T] tensor.
    - Keep patient-level context vectors unmodified (they don't need padding).

    Inputs:
        batch: list of items, each a dictionary with keys:
            - 'token_ids': LongTensor of shape [T_i] for each patient
            - 'time_deltas': FloatTensor of shape [T_i]
            - 'context_vec': FloatTensor of shape [C]
        pad_token_id: token index used to pad shorter sequences.

    Returns:
        A dictionary containing:
            - 'token_ids': LongTensor of shape [B, T_max] (padded)
            - 'time_deltas': FloatTensor of shape [B, T_max] (padded with 0.0)
            - 'context_vec': FloatTensor of shape [B, C]
    """
    batch_size = len(batch)
    max_len = max(len(x['token_ids']) for x in batch)

    padded_token_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_deltas = torch.zeros((batch_size, max_len), dtype=torch.float32)
    context_vectors = []

    for i, x in enumerate(batch):
        seq_len = len(x['token_ids'])
        padded_token_ids[i, :seq_len] = x['token_ids']
        padded_deltas[i, :seq_len] = x['time_deltas']
        context_vectors.append(x['context_vec'])

    context_vectors = torch.stack(context_vectors)
    return {
        'token_ids': padded_token_ids,
        'time_deltas': padded_deltas,
        'context_vec': context_vectors,
        'targets': padded_token_ids.clone()
    }