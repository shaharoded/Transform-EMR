import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ───────── local code ─────────────────────────────────────────────────── #
from config.dataset_config import *


class EMRDataset(Dataset):
    def __init__(self, df, patient_context_df, states, scaler=None):
        """
        df: original DataFrame with columns ['PatientID', 'ConceptName', 'StartDateTime', 'EndDateTime', 'Value']
        patient_context_df: DataFrame with columns ['PatientID'] + context_columns
        states: list of concepts to apply START/END tokenization to

        Attr:
            self.scaler (sklearn.preprocessing.StandardScaler): Used to scale the ctx vector. Can be inputed externally for validation/testing.
            self.context_df (pd.DataFrame): A non-temporal dataset for global context on the patient.
            self.tokens_df (pd.DataFrame): The temporal dataframe with the longtitudinal records of all the patients.
            self.patient_ids (pd.Series): Unique patient IDs in the data
            self.patient_groups (Dict): {pid: DataFrame} -> A sub-DataFrame containing the records of 1 patient as value
            self.token2id (Dict): token -> ID mapper.
        """
        self.context_df = patient_context_df.set_index("PatientID").astype("float32")
        
        # Fit scaler and transform context features
        self.scaler = scaler if scaler else StandardScaler()
        self.context_df.loc[:, :] = self.scaler.fit_transform(self.context_df.values)

        # Normalize time
        df['StartDateTime'] = pd.to_datetime(df['StartDateTime'], dayfirst=True)
        df['EndDateTime'] = pd.to_datetime(df['EndDateTime'], dayfirst=True)
        df['VisitStart'] = df.groupby('PatientID')['StartDateTime'].transform('min')
        df['RelStartTime'] = (df['StartDateTime'] - df['VisitStart']).dt.total_seconds() / 86400
        df['RelEndTime'] = (df['EndDateTime'] - df['VisitStart']).dt.total_seconds() / 86400

        # Expand tokens
        self.tokens_df = self._expand_tokens(df, states)

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

    def _expand_tokens(self, df, states):
        rows = []
        for _, row in df.iterrows():
            if row['ConceptName'] in states:
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
                token = row['ConceptName'] if row['Value'] == "True" else f"{row['ConceptName']}_{row['Value']}"
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
            'context_vector': context_vector
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
            - 'context_vector': FloatTensor of shape [C]
        pad_token_id: token index used to pad shorter sequences.

    Returns:
        A dictionary containing:
            - 'token_ids': LongTensor of shape [B, T_max] (padded)
            - 'time_deltas': FloatTensor of shape [B, T_max] (padded with 0.0)
            - 'context_vector': FloatTensor of shape [B, C]
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
        context_vectors.append(x['context_vector'])

    context_vectors = torch.stack(context_vectors)
    return {
        'token_ids': padded_token_ids,
        'time_deltas': padded_deltas,
        'context_vector': context_vectors
    }


if __name__ == "__main__":
    
    # Initiate dataset from files
    temporal_df = pd.read_csv(TEMPORAL_DATA_FILE)
    ctx_df = pd.read_csv(CTX_DATA_FILE)


    dataset = EMRDataset(df=temporal_df, patient_context_df=ctx_df, states=STATES)
    print('Number of Patients: ', len(dataset))
    print('Total Number of Records: ', len(dataset.tokens_df))
    print(dataset.tokens_df.head())

    # Get first patient's sample
    first_sample = dataset[0]
    print('First Patient Context Vector:', first_sample['context_vector'])