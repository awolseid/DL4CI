import pandas as pd
import numpy as np
import torch

from scripts.data_processing import prepare_denom_model_data
from scripts.models import LSTM4TreatmentExposureDenominatorModel
from scripts.training_and_testing import train_denom_model
from scripts.model_summaries import plot_training_metrics


def main():
    dir_path = "D://SimulatedData4CI//n_5000//SimGroup2"
    data_link = f"{dir_path}//data_alpha_0.2_gamma0.1_rho1_-0.5_rho2_0.4_n5000_rep1.csv"
    data_df = pd.read_csv(data_link)
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

    seed  = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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



if __name__ == "__main__":
    main()
