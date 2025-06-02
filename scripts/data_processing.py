import pandas as pd
import torch.nn as nn


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from scripts.helper_funs import determine_eligibility, subset_eligible_data
from scripts.tensor_datasets import LongitudinalDataset



def split_data(df              : pd.DataFrame,
               obs_id          : str,
               outcome         : str,
               test_size       : float         = 0.2,
               is_val_required : bool          = False):
    if is_val_required:
      print(f"\n     Data to be split into sets of train {(1-test_size)*100}%, validation {(test_size/2)*100}% and test {(test_size/2)*100}%.")
    else:
      print(f"     Data to be split into sets of train {(1-test_size)*100}% and test {(test_size)*100}%.")
    end_times_df = df.groupby(f"{obs_id}").tail(1).reset_index(drop=True)
    subj_outcome_df = end_times_df[[f"{obs_id}", f"{outcome}"]].drop_duplicates()
    train_ids, temp_ids = train_test_split(subj_outcome_df[f"{obs_id}"],
                                           stratify=subj_outcome_df[f"{outcome}"],
                                           test_size=test_size, random_state=42
                                        )
    train_df = df[df[f"{obs_id}"].isin(train_ids)].copy().reset_index(drop=True)

    if is_val_required:
        temp_df = subj_outcome_df[subj_outcome_df[f"{obs_id}"].isin(temp_ids)].reset_index(drop=True)
        val_ids, test_ids = train_test_split(temp_df[f"{obs_id}"],
                                             stratify=temp_df[f"{outcome}"],
                                             test_size=0.5, random_state=42
                                            )
        val_df   = df[df[f"{obs_id}"].isin(val_ids)].copy().reset_index(drop=True)
        test_df  = df[df[f"{obs_id}"].isin(test_ids)].copy().reset_index(drop=True)

    else:
        val_df  = None
        test_df = df[df[f"{obs_id}"].isin(temp_ids)].copy().reset_index(drop=True)
    print("     Spliting completed.")
    return train_df, val_df, test_df


def transform_fixed_covs(obs_id    : str,
                         train_df  : pd.DataFrame,
                         val_df    : pd.DataFrame  = None,
                         test_df   : pd.DataFrame  = None,
                         quan_covs : list[str]     = None,
                         qual_covs : list[str]     = None
                        ):

    def select_inv_obs(df):
        return df.sort_values(by=f"{obs_id}").groupby(f"{obs_id}").first().reset_index()

    inv_cols = [f"{obs_id}"] + (quan_covs or []) + (qual_covs or [])

    if (qual_covs is not None) or (quan_covs is not None):
        train_inv_df = select_inv_obs(train_df)[inv_cols]
    else:
        raise Exception("No variable to be encoded or scaled.")

    val_inv_df = select_inv_obs(val_df)[inv_cols] if val_df is not None else None
    test_inv_df = select_inv_obs(test_df)[inv_cols] if test_df is not None else None
    print(f"        Time-invariant features: {(quan_covs or []) + (qual_covs or [])}.")

    if quan_covs is not None:
        print("            Quantitative features to be scaled: ", quan_covs)
        scaler = StandardScaler()

        train_scaled_array = scaler.fit_transform(train_inv_df[quan_covs])
        scaled_names = [name + "_scaled" for name in
                        scaler.get_feature_names_out(quan_covs).tolist()]
        train_inv_df = train_inv_df.drop(columns=quan_covs)
        train_inv_df[scaled_names] = train_scaled_array

        if val_df is not None:
            val_scaled_array = scaler.transform(val_inv_df[quan_covs])
            val_inv_df = val_inv_df.drop(columns=quan_covs)
            val_inv_df[scaled_names] = val_scaled_array

        if test_df is not None:
            test_scaled_array = scaler.transform(test_inv_df[quan_covs])
            test_inv_df = test_inv_df.drop(columns=quan_covs)
            test_inv_df[scaled_names] = test_scaled_array
        print("            Standardized feature names: ", scaled_names)
    else:
        print("\n            No quantitative features to be scaled.")

    if qual_covs is not None:
        print("\n            Qualitative features to be encoded: ", qual_covs)
        encoder = OneHotEncoder(handle_unknown='ignore')
        train_encoded_array = encoder.fit_transform(train_inv_df[qual_covs]).toarray()
        encoded_names = encoder.get_feature_names_out(qual_covs).tolist()

        train_inv_df = train_inv_df.drop(columns=qual_covs)
        train_inv_df[encoded_names] = train_encoded_array
        if val_df is not None:
            val_encoded_array = encoder.transform(val_inv_df[qual_covs]).toarray()
            val_inv_df = val_inv_df.drop(columns=qual_covs)
            val_inv_df[encoded_names] = val_encoded_array

        if test_df is not None:
            test_encoded_array = encoder.transform(test_inv_df[qual_covs]).toarray()
            test_inv_df = test_inv_df.drop(columns=qual_covs)
            test_inv_df[encoded_names] = test_encoded_array
        print("            Encoded feature names: ", encoded_names)
    else:
        print("\n            No qualitative features to be encoded.")

    return train_inv_df, val_inv_df, test_inv_df

def transform_covs(obs_id    : str,
                   obs_time  : str,
                   outcome   : str,
                   train_df  : pd.DataFrame,
                   val_df    : pd.DataFrame  = None,
                   test_df   : pd.DataFrame  = None,
                   quan_covs : list[str]     = None,
                   qual_covs : list[str]     = None
                  ):
    if (qual_covs is None) and (quan_covs is None):
        raise Exception("No variable to be encoded or scaled.")
    else:
        print(f"\n        Time-varying features: {(quan_covs or []) + (qual_covs or [])}.")
    if quan_covs is not None:
        print("            Quantitative features to be scaled: ", quan_covs)
        scaler = StandardScaler()
        train_scaled_array = scaler.fit_transform(train_df[quan_covs])
        scaled_names = [name + "_scaled" for name in scaler.get_feature_names_out(quan_covs).tolist()]
        train_df[scaled_names] = train_scaled_array

        if val_df is not None:
            val_scaled_array = scaler.transform(val_df[quan_covs])
            val_df[scaled_names] = val_scaled_array
        if test_df is not None:
            test_scaled_array = scaler.transform(test_df[quan_covs])
            test_df[scaled_names] = test_scaled_array
        print("            Standardized feature names: ", scaled_names)
    else:
        print("\n            No quantitative features to be scaled.")

    if qual_covs is not None:
        print("\n            Qualitative features to be encoded: ", qual_covs)
        encoder = OneHotEncoder(handle_unknown='ignore')
        print(encoder, qual_covs)
        print(train_df.columns)
        train_encoded_array = encoder.fit_transform(train_df[qual_covs]).toarray()
        encoded_names = encoder.get_feature_names_out(qual_covs).tolist()
        train_df[encoded_names] = train_encoded_array
        if val_df is not None:
            val_encoded_array = encoder.transform(val_df[qual_covs]).toarray()
            val_df[encoded_names] = val_encoded_array
        if test_df is not None:
            test_encoded_array = encoder.transform(test_df[qual_covs]).toarray()
            test_df[encoded_names] = test_encoded_array
        print("            Encoded feature names: ", encoded_names)
    else:
        print("\n            No qualitative features to be encoded.")

    must_cols = ([f"{obs_id}"] + [f"{obs_time}"] + [f"{outcome}"] +
                 (scaled_names if quan_covs is not None else []) +
                 (encoded_names if qual_covs is not None else []))

    return (train_df[must_cols],
            val_df[must_cols] if val_df is not None else None,
            test_df[must_cols] if test_df is not None else None)

def preprocess_longitudinal_data(
    obs_id          : str,
    obs_time        : str,
    outcome         : str,
    train_df        : pd.DataFrame,
    val_df          : pd.DataFrame  = None,
    test_df         : pd.DataFrame  = None,
    quan_fixed_covs : list[str]     = None,
    quan_vary_covs  : list[str]     = None,
    qual_fixed_covs : list[str]     = None,
    qual_vary_covs  : list[str]     = None
    ):

    if quan_fixed_covs is not None or qual_fixed_covs is not None:
        train_fixed_df, val_fixed_df, test_fixed_df = transform_fixed_covs(obs_id = obs_id,
                                                                           train_df = train_df,
                                                                           val_df = val_df,
                                                                           test_df = test_df,
                                                                           quan_covs = quan_fixed_covs,
                                                                           qual_covs = qual_fixed_covs)
    else:
        train_fixed_df = val_fixed_df = test_fixed_df = None

    if quan_vary_covs is not None or qual_vary_covs is not None:
        train_vary_df, val_vary_df, test_vary_df = transform_covs(obs_id    = obs_id,
                                                                  obs_time  = obs_time,
                                                                  outcome   = outcome,
                                                                  train_df  = train_df,
                                                                  val_df    = val_df,
                                                                  test_df   = test_df,
                                                                  quan_covs = quan_vary_covs,
                                                                  qual_covs = qual_vary_covs)
    else:
        train_vary_df = val_vary_df = test_vary_df = None

    if train_vary_df is not None and train_fixed_df is not None:
        train_scaled_df = pd.merge(train_vary_df, train_fixed_df, on=obs_id, how='inner')
    elif train_vary_df is not None and train_fixed_df is None:
        train_scaled_df = train_vary_df
    elif train_vary_df is None and train_fixed_df is not None:
        train_scaled_df = train_fixed_df
    else:
        train_scaled_df = None

    if val_vary_df is not None and val_fixed_df is not None:
        val_scaled_df = pd.merge(val_vary_df, val_fixed_df, on=obs_id, how='inner')
    elif val_vary_df is not None and val_fixed_df is None:
        val_scaled_df = val_vary_df
    elif val_vary_df is None and val_fixed_df is not None:
        val_scaled_df = val_fixed_df
    else:
        val_scaled_df = None

    if test_vary_df is not None and test_fixed_df is not None:
        test_scaled_df = pd.merge(test_vary_df, test_fixed_df, on=obs_id, how='inner')
    elif test_vary_df is not None and test_fixed_df is None:
        test_scaled_df = test_vary_df
    elif test_vary_df is None and test_fixed_df is not None:
        test_scaled_df = test_fixed_df
    else:
        test_scaled_df = None

    return train_scaled_df, val_scaled_df, test_scaled_df


def get_transformed_feature_names(transformed_df   : pd.DataFrame,
                                  qual_fixed_covs  : list[str]     = None,
                                  qual_vary_covs   : list[str]     = None,
                                  quan_fixed_covs  : list[str]     = None,
                                  quan_vary_covs   : list[str]     = None
                                 ):
    encoded_fixed_names = [col_name for col_name in transformed_df.columns
                     if any(col_name.startswith(f"{var}_") for var in
                            (qual_fixed_covs or []))]
    encoded_vary_names = [col_name for col_name in transformed_df.columns
                         if any(col_name.startswith(f"{var}_") for var in
                                (qual_vary_covs or []))]

    scaled_fixed_names = [col_name for col_name in transformed_df.columns
                         if any(col_name.startswith(f"{var}_") for var in
                                (quan_fixed_covs or []))]
    scaled_vary_names = [col_name for col_name in transformed_df.columns
                         if any(col_name.startswith(f"{var}_") for var in
                                (quan_vary_covs or []))]
    return {"fixed_feature_names": encoded_fixed_names + scaled_fixed_names,
            "vary_feature_names": encoded_vary_names + scaled_vary_names}

def prepare_sequences(df                  : pd.DataFrame,
                      obs_id              : str,
                      obs_time            : str,
                      outcome             : str,
                      fixed_feature_names : str,
                      vary_feature_names  : str):
    print(f"\n     Started preparing sequence data...")
    seq_data = []
    for _, group in df.groupby(obs_id):
        group = group.sort_values(obs_time)
        data = {'static': group[fixed_feature_names].iloc[0].values,
                'longitudinal': group[vary_feature_names].values,  # shape: (T, D)
                'labels': group[outcome].values  # shape: (T,)
               }
        seq_data.append(data)
    print("     Sequence data preparation completed.")
    return seq_data


def prepare_denom_model_data(df              : pd.DataFrame,
                        obs_id          : str,
                        obs_time        : str,
                        event           : str,
                        outcome         : str,
                        quan_fixed_covs : list[str],
                        qual_fixed_covs : list[str],
                        quan_vary_covs  : list[str],
                        qual_vary_covs  : list[str],
                        is_val_required : bool          = False
                        ):
    df      = df.sort_values([f"{obs_id}", f"{obs_time}"])
    df      = determine_eligibility(df, obs_id, obs_time, outcome)
    elig_df = subset_eligible_data(df, obs_id, obs_time, outcome)
    trn_df, vld_df, tst_df = split_data(elig_df, obs_id, outcome,
                                        is_val_required=is_val_required)

    trn_scaled_df, vld_scaled_df, tst_scaled_df = preprocess_longitudinal_data(
                                                    obs_id          = obs_id,
                                                    obs_time        = obs_time,
                                                    outcome         = outcome,
                                                    train_df        = trn_df,
                                                    val_df          = vld_df,
                                                    test_df         = tst_df,
                                                    quan_fixed_covs = quan_fixed_covs,
                                                    quan_vary_covs  = quan_vary_covs,
                                                    qual_fixed_covs = qual_fixed_covs,
                                                    qual_vary_covs  = qual_vary_covs)

    feature_names = get_transformed_feature_names(transformed_df   = trn_scaled_df,
                                                    qual_fixed_covs  = qual_fixed_covs,
                                                    qual_vary_covs   = qual_vary_covs,
                                                    quan_fixed_covs  = quan_fixed_covs,
                                                    quan_vary_covs   = quan_vary_covs
                                                    )

    fixed_feature_names = feature_names["fixed_feature_names"]
    vary_feature_names  = feature_names["vary_feature_names"]

    train_seq_data = prepare_sequences(trn_scaled_df,
                                        obs_id,
                                        obs_time,
                                        outcome,
                                        fixed_feature_names,
                                        vary_feature_names)

    if vld_scaled_df is not None:
        val_seq_data = prepare_sequences(vld_scaled_df,
                                        obs_id,
                                        obs_time,
                                        outcome,
                                        fixed_feature_names,
                                        vary_feature_names)
    else:
        val_seq_data = None

    test_seq_data = prepare_sequences(tst_scaled_df,
                                        obs_id,
                                        obs_time,
                                        outcome,
                                        fixed_feature_names,
                                        vary_feature_names)

    train_dataset = LongitudinalDataset(train_seq_data)
    if val_seq_data is not None:
        val_dataset = LongitudinalDataset(val_seq_data)
    else:
        val_dataset = None
    test_dataset = LongitudinalDataset(test_seq_data)

    return train_dataset, val_dataset, test_dataset



def extract_baseline_vars(df        : pd.DataFrame,
                          obs_id    : str,
                          outcome   : str,
                          qual_covs : list[str]     = None,
                          quan_covs : list[str]     = None
                          ):
  print("\n-----Preparing to extract baseline baseline variables ...")
  features    = [f"{obs_id}"] + (qual_covs or []) + (quan_covs or []) + [f"{outcome}"]
  idx         = df.groupby(f"{obs_id}")[f"{obs_time}"].idxmin()
  baseline_df = df.loc[idx, features].reset_index(drop=True)
  print("\n-----Extracting baseline variables completed!")
    
  return baseline_df