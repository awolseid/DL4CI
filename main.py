import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.helper_funs import set_seed, collate_fn
from scripts.data_processing import (
    prepare_denom_model_data, extract_baseline_vars, split_data)
from scripts.model_archs import (
    LSTM4TreatmentExposureDenominatorModel, TreatmentExposureNumeratorModel)
from scripts.training_and_testing import (
    train_denom_model, GetDenomModelPredictions,
    train_numer_model, test_numer_model)
from scripts.cross_validation import (
    grid_search_denom_model, grid_search_cv_numer_model)
from scripts.model_summaries import plot_training_metrics
from scripts.tensor_datasets import get_tensor_datasets

set_seed()

dir_path = Path("D:/SimulatedData4CI/n_1000")

for subdir in dir_path.iterdir():

    optimal_hyperparameters= {}

    if subdir.is_dir():
 
        subdir_path = subdir.resolve()
        first_file = sorted(os.listdir(subdir_path))[0]
        data_path = subdir_path/first_file
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


        print("K-CV for the Denominator model started ....")
        denom_param_grid = {
            'hidden_dim' : [16], #, 32, 64, 128],
            'lr'         : [0.01],#, 0.001, 0.0001],
            'batch_size' : [16],#, 32, 64, 128],
            'epochs'     : [1],
        }

        train_dataset, val_dataset, test_dataset = prepare_denom_model_data(df              = data_df,
                                                                            obs_id          = obs_id,
                                                                            obs_time        = obs_time,
                                                                            event           = event,
                                                                            outcome         = outcome,
                                                                            quan_fixed_covs = quan_fixed_covs,
                                                                            qual_fixed_covs = qual_fixed_covs,
                                                                            quan_vary_covs  = quan_vary_covs,
                                                                            qual_vary_covs  = qual_vary_covs,
                                                                            is_val_required = False
                                                                            )


        denom_best_params, denom_best_summaries, denom_best_loss = grid_search_denom_model(
                                                model_class = LSTM4TreatmentExposureDenominatorModel,
                                                dataset     = train_dataset,
                                                param_grid  = denom_param_grid,
                                                collate_fn  = collate_fn
        )
        optimal_hyperparameters[f"{first_file}_denom"] = denom_best_params

        print("Optimal hyperparameters for Denom Model obtained.")

        print("\nTraining Denominator Model with optimal hyperparameters ...")
        bs     = denom_best_params['batch_size']
        hd     = denom_best_params['hidden_dim']
        lr     = denom_best_params['lr']
        epochs = denom_best_params['epochs']

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
        # plot_training_metrics(treat_denom_metrics)
        test_loader= DataLoader(dataset = test_dataset, batch_size = 16, collate_fn = collate_fn)
        GetDenomModelPredictions().test_denom_model(
            trained_model = treat_denom_model, data_loader   = test_loader
            )
        print(f"Denominator Model with optimal hyperparameters training completed!")




        print("\nK-CV for the Numerator model started ....")
        qual_covs = None
        quan_covs = ["X1", "X2"]
        numer_param_grid = {
            "hidden_dim" : [16], #, 32, 64],
            "lr"         : [0.01],#, 0.001, 0.0001],
            "batch_size" : [16],# 32, 64, 128],
            'epochs'     : [1],
        }

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
                                                        is_val_required = False
                                                    )

        train_ds, val_ds, test_ds = get_tensor_datasets(obs_id = obs_id,
                                                        outcome = outcome,
                                                        train_raw_df = trn_raw_df, 
                                                        val_raw_df = vld_raw_df, 
                                                        test_raw_df  = tst_raw_df, 
                                                        qual_covs    = qual_covs, 
                                                        quan_covs   = quan_covs
                                                        )

        numer_best_params, numer_best_summaries, numer_best_loss = grid_search_cv_numer_model(
                                                model_class = TreatmentExposureNumeratorModel,
                                                dataset     = train_ds,
                                                param_grid  = numer_param_grid
                                                )
        optimal_hyperparameters[f"{first_file}_numer"] = numer_best_params

        print("Optimal hyperparameters for Numer Model obtained.")




        print("\nTraining Numerator model with optimal hyperparameters ...")
        bs     = numer_best_params['batch_size']
        hd     = numer_best_params['hidden_dim']
        lr     = numer_best_params['lr']
        epochs = numer_best_params['epochs']

        treat_numer_model   = TreatmentExposureNumeratorModel(
                            input_dim  = len(list(train_ds)[0][0]), 
                            hidden_dim = hd)

        treat_numer_metrics = train_numer_model(model         = treat_numer_model, 
                                                train_dataset = train_ds, 
                                                val_dataset   = val_ds,
                                                batch_size    = bs,
                                                lr            = lr,
                                                epochs        = epochs
                                            )

        # plot_training_metrics(treat_numer_metrics)

        test_numer_model(treat_numer_model, test_loader = DataLoader(test_ds, batch_size = bs))
        print(f"\nNumerator Model  with optimal hyperparameters training completed!")




        optimal_hyperparams_df = pd.DataFrame(optimal_hyperparameters).T

        saving_folder_name = subdir_path.parent.name
        saving_file_name   = subdir_path.name + "Optimal_Hyperparams"
        saving_dir         = Path("results") / saving_folder_name

        saving_dir.mkdir(parents=True, exist_ok=True)

        optimal_hyperparams_df.to_csv(f"{saving_dir}/{saving_file_name}.csv")



