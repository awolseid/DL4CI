import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class LongitudinalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


def get_tensor_datasets(obs_id       : str,
                        outcome      : str,
                        train_raw_df : pd.DataFrame,  
                        test_raw_df  : pd.DataFrame, 
                        qual_covs    : list[str], 
                        quan_covs    : list[str],
                        val_raw_df   : pd.DataFrame = None
                        ):
  print("\n-----Converting raw df to tensor datasets ...!")  
  y_trn    = train_raw_df[f"{outcome}"]
  X_trn_df = train_raw_df.drop(columns=[f"{obs_id}", f"{outcome}"])

  if val_raw_df is not None:
    y_vld    = val_raw_df[f"{outcome}"]
    X_vld_df = val_raw_df.drop(columns=[f"{obs_id}", f"{outcome}"])
  else:
    y_vld    = None
    X_vld_df = None

  y_tst    = test_raw_df[f"{outcome}"]
  X_tst_df = test_raw_df.drop(columns=[f"{outcome}"])

  numeric_transformer = StandardScaler()
  onehot_transformer = OneHotEncoder()

  if qual_covs is not None and quan_covs is None:
      transformer = ColumnTransformer(
          [("OneHotEncoder", onehot_transformer, qual_covs)]
      )
  elif qual_covs is None and quan_covs is not None:
      transformer = ColumnTransformer(
          [("StandardScaler", numeric_transformer, quan_covs)]
      )
  elif qual_covs is not None and quan_covs is not None:
      transformer = ColumnTransformer(
           [("OneHotEncoder", onehot_transformer, qual_covs),
            ("StandardScaler", numeric_transformer, quan_covs)]
       )
  else:
       raise ValueError("At least one of qual_covs or quan_covs must NOT be None.")
  print(transformer)

  X_trn_tensor     = torch.tensor(transformer.fit_transform(X_trn_df), dtype=torch.float32)
  if X_vld_df is not None:
    X_vld_tensor   = torch.tensor(transformer.transform(X_vld_df), dtype=torch.float32)
  else: 
      X_vld_tensor = None
  X_tst_tensor     = torch.tensor(transformer.transform(X_tst_df), dtype=torch.float32)

  y_trn_tensor     = torch.tensor(np.array(y_trn), dtype=torch.float32).unsqueeze(1)
  if y_vld is not None:
      y_vld_tensor = torch.tensor(np.array(y_vld), dtype=torch.float32)
  else: 
      y_vld_tensor = None
  y_tst_tensor     = torch.tensor(np.array(y_tst), dtype=torch.float32).unsqueeze(1)

  trn_ds     = TensorDataset(X_trn_tensor, y_trn_tensor)
  if X_vld_tensor is not None: 
      vld_ds = TensorDataset(X_vld_tensor, y_vld_tensor)
  else: 
      vld_ds = None
  tst_ds     = TensorDataset(X_tst_tensor, y_tst_tensor)
  print("\n-----Tensor datasets obtained!")  
    
  return trn_ds, vld_ds, tst_ds