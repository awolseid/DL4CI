import os
import pandas as pd
import numpy as np
import torch

from scripts.helper_funs import set_seed
from scripts.data_processing import extract_baseline_vars, split_data
from scripts.tensor_datasets import get_tensor_datasets
from scripts.model_archs import TreatmentExposureNumeratorModel
from scripts.training_and_testing import train_numer_model
from scripts.model_summaries import plot_training_metrics



set_seed()

dir_path = "D://SimulatedData4CI//n_1000//SimGroup1"
first_file = sorted(os.listdir(dir_path))[0]
data_path = os.path.join(dir_path, first_file)

data_df = pd.read_csv(data_path)
print(data_df)
print("\nData is successfully imported.")


outcome    = "Treatment"
obs_id     = "ID"
obs_time   = "Obs_Time"
qual_covs = ["X2"]
quan_covs = ["X1"]

invariant_df = extract_baseline_vars(df        = data_df,
                                     obs_id    = obs_id,
                                     obs_time  = obs_time,
                                     outcome   = outcome,
                                     qual_covs = qual_covs,
                                     quan_covs = quan_covs
                                     )


trn_raw_df, vld_raw_df, tst_raw_df = split_data(df              = invariant_df, 
                                                obs_id          = obs_id, 
                                                outcome         = outcome,
                                                is_val_required = True
                                               )
train_ds, val_ds, test_ds = get_tensor_datasets(obs_id       = obs_id,
                                                outcome      = outcome,
                                                train_raw_df = trn_raw_df, 
                                                val_raw_df   = vld_raw_df, 
                                                test_raw_df  = tst_raw_df, 
                                                qual_covs    = qual_covs, 
                                                quan_covs    = quan_covs
)


bs     = 16
hd     = 16
lr     = 0.001
epochs = 200



treat_numer_model   = TreatmentExposureNumeratorModel(
                      input_dim  = len(list(train_ds)[0][0]), 
                      hidden_dim = hd
                    )

treat_numer_metrics = train_numer_model(treat_numer_model, 
                                        train_dataset      = train_ds, 
                                        val_dataset        = val_ds,
                                        batch_size         = bs,
                                        epochs             = epochs, 
                                        lr                 = lr)
plot_training_metrics(treat_numer_metrics)
