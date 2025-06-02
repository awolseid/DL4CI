def save_IPWs(survival_df  : pd.DataFrame,
              ipw_df       : pd.DataFrame,
              subj_id_col  : str,
              time_id_col  : str,
              ):
    if len(survival_df) != len(ipw_df):
        raise Exception(f"Lengths of {survival_df} and {ipw_df} are different!")
    survival_df = survival_df.sort_values([f"{subj_id_col}", f"{time_id_col}"])
    ipw_df      = ipw_df.sort_values([f"{subj_id_col}", f"{time_id_col}"])
    weighted_df = pd.merge(survival_df, ipw_df, on=[f"{subj_id_col}",f"{time_id_col}"], how='inner')
    print("\n-----IPW or probabilties included in the data frame.")
    return weighted_df





def save_denom_IPWs(dir_path : str, treat_denom_model : nn.Module, treat_numer_model : nn.Module):
    num_files = 0
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path      = os.path.join(dir_path, filename)
            data_df        = pd.read_csv(file_path)
            
            denom_treat_df = get_denom_model_summary(treat_denom_model,
                                                     data_df,
                                                     obs_id,
                                                     obs_time,
                                                     outcome,
                                                     outcome_type        = "exposure",
                                                     quan_fixed_covs     = quan_fixed_covs,
                                                     quan_vary_covs      = quan_vary_covs,
                                                     qual_fixed_covs     = qual_fixed_covs,
                                                     qual_vary_covs      = qual_vary_covs
                                                    )
            
            weights_df = save_IPWs(survival_df  = data_df,
                                   ipw_df       = denom_treat_df,
                                   subj_id_col  = obs_id,
                                   time_id_col  = obs_time
                                   )
            
            treat_numer_df = get_numer_model_summary(treat_numer_model,
                                                     data_df,
                                                     obs_id              = obs_id,
                                                     obs_time            = obs_time,
                                                     outcome             = outcome,
                                                     outcome_type        = "exposure",
                                                     quan_covs           = quan_covs,
                                                     qual_covs           = qual_covs
                                                    )
            
            weights_df = save_IPWs(survival_df  = weights_df,
                                   ipw_df       = treat_numer_df,
                                   subj_id_col  = obs_id,
                                   time_id_col  = obs_time)

            weights_df["StabilizedIPTWs"] = weights_df["NumerTreatProbs"] / weights_df["DenomTreatProbs"]            
    
            weights_df.to_csv(file_path, index = False)
            
        num_files += 1  
            
    print(f"Saving {num_files} files completed!")
