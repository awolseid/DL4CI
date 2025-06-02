import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn


def get_denom_model_summary(trained_model   : nn.Module,
                            df              : pd.DataFrame,
                            obs_id          : str,
                            obs_time        : str,
                            outcome         : str,
                            outcome_type    : str           = "exposure",
                            quan_fixed_covs : list[str]     = None,
                            quan_vary_covs  : list[str]     = None,
                            qual_fixed_covs : list[str]     = None,
                            qual_vary_covs  : list[str]     = None
                            ):

    print("-----Data processing started...")
    scaled_df, _, _ = preprocess_longitudinal_data(obs_id          = obs_id,
                                                   train_df        = df,
                                                   quan_fixed_covs = quan_fixed_covs,
                                                   quan_vary_covs  = quan_vary_covs,
                                                   qual_fixed_covs = qual_fixed_covs,
                                                   qual_vary_covs  = qual_vary_covs
                                                   )
    print("-----Data processing completed.")
    print("\n-----Converting data to longitudinal sequences ...")
    feat_names = get_transformed_feature_names(transformed_df  = scaled_df,
                                               qual_fixed_covs = qual_fixed_covs,
                                               qual_vary_covs  = qual_vary_covs,
                                               quan_fixed_covs = quan_fixed_covs,
                                               quan_vary_covs  = quan_vary_covs
                                               )

    fixed_feature_names = feat_names["fixed_feature_names"]
    vary_feature_names  = feat_names["vary_feature_names"]

    seq_data = prepare_sequences(scaled_df,
                                 obs_id,
                                 obs_time,
                                 outcome,
                                 fixed_feature_names,
                                 vary_feature_names)
    print("-----Converting data to longitudinal sequences completed.")
    print("\n-----Creating tensor datasets and dataloaders started...")
    dataset = LongitudinalDataset(seq_data)
    data_loader= DataLoader(dataset, batch_size = 16, collate_fn=collate_fn)
    print("-----Creating tensor datasets and dataloaders completed.")
    print("\n-----Getting predicted probabilities using trained model...")
    model_predictions = GetDenomModelPredictions()
    model_predictions.test_denom_model(trained_model, data_loader)
    pred_probs = model_predictions.pred_probs
    print("-----Getting predicted probabilities using trained model completed.")
    print("\n-----IPW probabilties are being obtained...")
    df = determine_eligibility(df, obs_id, obs_time, outcome)
    IPW_df = df.copy()[[f"{obs_id}", f"{obs_time}", f"{outcome}", "is_eligible"]]

    condition1 = IPW_df['is_eligible'] == 0
    condition2 = (IPW_df['is_eligible'] == 1) & (IPW_df[f'{outcome}'] == 1)
    condition3 = (IPW_df['is_eligible'] == 1) & (IPW_df[f'{outcome}'] == 0)

    print("\n-----IPW probabilties being obtained.")
    if outcome_type == "exposure":
        IPW_df["DenomP_it(E)"] = pred_probs
        IPW_df["DenomP_it(E=e)"] = np.nan

        IPW_df.loc[condition1, 'DenomP_it(E=e)'] = 1
        IPW_df.loc[condition2, 'DenomP_it(E=e)'] = IPW_df.loc[IPW_df[f'{outcome}'] == 1, 'DenomP_it(E)']
        IPW_df.loc[condition3, 'DenomP_it(E=e)'] = 1 - IPW_df.loc[IPW_df[f'{outcome}'] == 0, 'DenomP_it(E)']

        IPW_df['DenomTreatProbs'] = IPW_df.groupby(f'{obs_id}')['DenomP_it(E=e)'].cumprod()
        IPW_df["UnstabilizedIPTWs"] = 1 / IPW_df['DenomTreatProbs']
        print(f"\nSummary of 'UnstabilizedIPTWs': \n{IPW_df["UnstabilizedIPTWs"].describe()}")
    elif outcome_type == "censor":
        IPW_df["DenomP_it(C)"] = pred_probs
        IPW_df["DenomP_it(C=1)"] = np.nan
        IPW_df.loc[condition1, 'DenomP_it(C=1)'] = 1
        IPW_df.loc[condition2, 'DenomP_it(C=1)'] = IPW_df.loc[IPW_df[f'{outcome}'] == 1, 'DenomP_it(C)']
        IPW_df.loc[condition3, 'DenomP_it(C=1)'] = IPW_df.loc[IPW_df[f'{outcome}'] == 0, 'DenomP_it(C)']

        IPW_df['DenomCensorProbs'] = IPW_df.groupby(f'{obs_id}')['DenomP_it(C=1)'].cumprod()
        IPW_df["UnstabilizedIPCWs"] = 1 / IPW_df['DenomCensorProbs']
        print(f"\nSummary of 'UnstabilizedIPCWs': \n{IPW_df["UnstabilizedIPCWs"].describe()}")
    else:
        raise Exception("Outcome type must be 'exposure' for treatment or 'censor' for censoring!")
    print("\n-----IPW probabilties obtained.")
    return IPW_df.drop(columns=[f"{outcome}", "is_eligible"])



def plot_training_metrics(metrics):
    epochs_range = range(1, len(metrics["train_losses"]) + 1)
    plt.figure(figsize = (8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, metrics["train_losses"], label='Train')
    if metrics["val_losses"] != []:
        plt.plot(epochs_range, metrics["val_losses"], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, metrics["train_accuracies"], label='Train', color='darkorange')
    if metrics["val_accuracies"] != []:
        plt.plot(epochs_range, metrics["val_accuracies"], label='Validation', color='green')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



def get_numer_model_summary(trained_model   : nn.Module,
                            df              : pd.DataFrame,
                            obs_id          : str,
                            obs_time        : str,
                            outcome         : str,
                            outcome_type    : str           = "exposure",
                            quan_covs       : list[str]     = None,
                            qual_covs       : list[str]     = None,
                            ):

    print("-----Data processing started...")
    scaled_df, _, _ = transform_covs(obs_id    = obs_id,
                                     obs_time  = obs_time,
                                     outcome   = outcome,
                                     train_df  = df,
                                     quan_covs = quan_covs,
                                     qual_covs = qual_covs)

    print("-----Data processing completed.")
    print(df.shape, scaled_df.shape)
    print("\n-----Creating tensor datasets and dataloaders started...")
    y = scaled_df[f"{outcome}"]

    encoded_names = [col_name for col_name in scaled_df.columns
                     if any(col_name.startswith(f"{var}_") for var in (qual_covs or []))]
    scaled_names = [col_name for col_name in scaled_df.columns
                    if any(col_name.startswith(f"{var}_") for var in (quan_covs or []))]
    X_df = scaled_df[encoded_names + scaled_names]

    X_tensor    = torch.tensor(np.array(X_df), dtype=torch.float32)
    y_tensor    = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    dataset     = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=16)
    print("-----Creating tensor datasets and dataloaders completed.")
    print("\n-----Getting predicted probabilities using trained model...")
    pred_probs, _ , _ = test_numer_model(trained_model, data_loader)
    print("-----Getting predicted probabilities using trained model completed.")
    print("\n-----IPW probabilties are being obtained...")
    df = determine_eligibility(df, obs_id, obs_time, outcome)
    IPW_df = df.copy()[[f"{obs_id}", f"{obs_time}", f"{outcome}", "is_eligible"]]

    condition1 = IPW_df['is_eligible'] == 0
    condition2 = (IPW_df['is_eligible'] == 1) & (IPW_df[f'{outcome}'] == 1)
    condition3 = (IPW_df['is_eligible'] == 1) & (IPW_df[f'{outcome}'] == 0)

    if outcome_type == "exposure":
        IPW_df["P_it(E)"] = pred_probs
        IPW_df["P_it(E=e)"] = np.nan

        IPW_df.loc[condition1, 'P_it(E=e)'] = 1
        IPW_df.loc[condition2, 'P_it(E=e)'] = IPW_df.loc[IPW_df[f'{outcome}'] == 1, 'P_it(E)']
        IPW_df.loc[condition3, 'P_it(E=e)'] = 1 - IPW_df.loc[IPW_df[f'{outcome}'] == 0, 'P_it(E)']

        IPW_df['NumerTreatProbs'] = IPW_df.groupby(f'{obs_id}')['P_it(E=e)'].cumprod()
        print(f"\nSummary of 'NumerTreatProbs': \n{IPW_df["NumerTreatProbs"].describe()}")
    elif outcome_type == "censor":
        IPW_df["P_it(C)"] = pred_probs
        IPW_df["P_it(C=1)"] = np.nan
        IPW_df.loc[condition1, 'P_it(C=1)'] = 1
        IPW_df.loc[condition2, 'P_it(C=1)'] = IPW_df.loc[IPW_df[f'{outcome}'] == 1, 'P_it(C)']
        IPW_df.loc[condition3, 'P_it(C=1)'] = IPW_df.loc[IPW_df[f'{outcome}'] == 0, 'P_it(C)']

        IPW_df['NumerCensorProbs'] = IPW_df.groupby(f'{obs_id}')['P_it(C=1)'].cumprod()
        print(f"\nSummary of 'NumerCensorProbs': \n{IPW_df["NumerCensorProbs"].describe()}")
    else:
        raise Exception("Outcome type must be 'exposure' for treatment or 'censor' for censoring!")
    print("-----IPW probabilties obtained.")
    return IPW_df.drop(columns=[f"{outcome}", "is_eligible"])    