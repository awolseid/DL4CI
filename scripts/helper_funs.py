import pandas as pd
import numpy as np
import random

import torch
from torch.nn.utils.rnn import pad_sequence

def determine_eligibility(df: pd.DataFrame, obs_id: str, obs_time: str, outcome: str):
    def eligibility_group(outcome_series):
        outcome_list = outcome_series.tolist()
        if 1 in outcome_list:
            match_index = outcome_list.index(1)
            return [1] * (match_index + 1) + [0] * (len(outcome_list) - match_index - 1)
        else:
            return [1] * len(outcome_list)
    df = df.copy()
    df.sort_values(by=[f"{obs_id}", f"{obs_time}"], ascending=[True, True], inplace=True)
    df['is_eligible'] = df.groupby(f"{obs_id}")[f"{outcome}"].transform(lambda x: eligibility_group(x))
    print(f"\n     An 'is_eligible' variable is included to identify observations for modeling '{outcome}'.")
    print(f"     Of {len(df)} observations, {sum(df['is_eligible']==1)} are eligible.")
    return df

def subset_eligible_data(df: pd.DataFrame, obs_id: str, obs_time: str, outcome: str):
    df1 = df.copy().sort_values([f"{obs_id}", f"{obs_time}"])
    df1 = determine_eligibility(df1, obs_id, obs_time, outcome)
    elig_df = df1[df1["is_eligible"] == 1].copy().reset_index(drop=True)
    print(f"\n     A dataframe containing only {len(elig_df)} eligible observations for IPW model is extracted.")
    del df1
    return elig_df


def collate_fn(batch):
    static = torch.tensor(np.array([item['static'] for item in batch]), dtype=torch.float32)
    longit = [torch.tensor(item['longitudinal'], dtype=torch.float32) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.float32) for item in batch]

    padded_longit = pad_sequence(longit, batch_first=True, padding_value=-1.0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1.0)

    return static, padded_longit, padded_labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)            # for single GPU
    torch.cuda.manual_seed_all(seed)        # for multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
