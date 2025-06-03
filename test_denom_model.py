import os
import pandas as pd
import numpy as np
import torch

from scripts.helper_funs import set_seed
from scripts.data_processing import prepare_denom_model_data
from scripts.model_archs import LSTM4TreatmentExposureDenominatorModel
from scripts.training_and_testing import train_denom_model
from scripts.model_summaries import plot_training_metrics
from scripts.cross_validation import grid_search_denom_model



set_seed()

dir_path = "D://SimulatedData4CI//n_1000//SimGroup1"
first_file = sorted(os.listdir(dir_path))[0]
data_path = os.path.join(dir_path, first_file)

data_df = pd.read_csv(data_path)
print(data_df)
print("\nData is successfully imported.")

event      = "Event"
outcome    = "Treatment"
obs_id     = "ID"
obs_time   = "Obs_Time"

quan_fixed_covs = ["X1"]
qual_fixed_covs = ["X2"]
quan_vary_covs = ["Obs_Time", "Confounder1"]
qual_vary_covs = ["Confounder2"]



train_dataset, val_dataset, test_dataset = prepare_denom_model_data(df              = data_df,
                                                                    obs_id          = obs_id,
                                                                    obs_time        = obs_time,
                                                                    event           = event,
                                                                    outcome         = outcome,
                                                                    quan_fixed_covs = quan_fixed_covs,
                                                                    qual_fixed_covs = qual_fixed_covs,
                                                                    quan_vary_covs  = quan_vary_covs,
                                                                    qual_vary_covs  = qual_vary_covs,
                                                                    is_val_required = True
                                                                    )

bs     = 64
hd     = 128
lr     = 0.001
epochs = 50



treat_denom_model = LSTM4TreatmentExposureDenominatorModel(
                        longit_input_dim = train_dataset[0]["longitudinal"].shape[1],
                        static_input_dim = len(train_dataset[0]["static"]),
                        hidden_dim       = hd)

print(f"\nModel architecture:")
print(f"\n")
print(f"\n{treat_denom_model}")
print(f"\n")
print(f"\nModel training started...")


treat_denom_metrics = train_denom_model(model         = treat_denom_model,
                                        train_dataset = train_dataset,
                                        val_dataset   = val_dataset,
                                        lr            = lr,
                                        batch_size    = bs,
                                        epochs        = epochs
                                        )
print(f"\nModel training completed!")
plot_training_metrics(treat_denom_metrics)