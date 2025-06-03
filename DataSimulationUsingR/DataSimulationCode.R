library(data.table)
library(copula)
library(tidyr)
library(dplyr)

sigmoid <- function(x) {1 / (1 + exp(-x))}

#set.seed(42)

simulate_causal_inference_data <- function(alpha, # Coeff of treatment in Surv model
                                           beta,  # Coeff of X_2 in Surv model
                                           gamma, # Coeff of treatment*X_2 in Surv model
                                           rho1, 
                                           rho2,
                                           K,
                                           n
                                           ) 
  {
  # copulas
  copulaL1 <- tCopula(param   = rho1, 
                      df      = 5, 
                      dim     = 2, 
                      dispstr = "un")
  
  copulaL2 <- normalCopula(param = rho2, 
                           dim = 2, 
                           dispstr = "un")
  
  # Data table for storing simulated data
  sim_df      <- data.table()
  sim_df$ID   <- 1 : n
  sim_df$Status <- rep(1, n)
  
  # Baseline covariates
  sim_df[, X1 := rexp(n, rate = 2)]
  sim_df[, X2 := rbinom(n, size = 1, prob = 0.7)]
  
  base_covs <- c("X1", "X2")
  
  # Data table for quantiles to be carried over to the next time step
  quant_df <- copy(sim_df[, base_covs, with = FALSE])
  
  for (k in 0:K) {
    n_Status <- sum(sim_df$Status, na.rm = TRUE)
    
    if (k > 0) {
      quant_df <- quant_df[Event == 0, ]
      
      if (k > 1) {
        quant_df[, grep("_prev$", colnames(quant_df), value = TRUE) := NULL]
      }
      
      # reset names
      setnames(quant_df, old = setdiff(names(quant_df), base_covs),
               new = paste0(setdiff(names(quant_df), base_covs), "_prev"))
    }
    
    # Random quantiles
    quant_df[, paste0("F_L1",     k, "|Status")              := runif(n_Status)]
    quant_df[, paste0("F_L2",     k, "|L1"   , k, "_Status") := runif(n_Status)]
    quant_df[, paste0("F_E",      k, "|L1"   , k, "_Status") := runif(n_Status)]
    quant_df[, paste0("F_Event|L1L2", k, "_Status")              := runif(n_Status)]
    
    if (k == 0) {
      quant_df[, L1 := qgamma(get(paste0("F_L1", k, "|Status")),
                              shape = 1 + 0.5*X1 + 0.5*X2,
                              scale = 1)]
      quant_df[, L2 := qbinom(get(paste0("F_L2", k, "|L1", k, "_Status")),
                              size = 1,
                              prob = sigmoid(-0.2 + 0.5*X1 + 0.5*X2))]
    } 
    else {
      quant_df[, L1 := qgamma(get(paste0("F_L1", k, "|Status")),
                       shape = 1 + 0.5*X1 + 0.5*X2 + 0.1*L1_prev - 0.5*E_prev, 
                       scale = 1)]
      quant_df[, L2 := qbinom(get(paste0("F_L2", k, "|L1", k, "_Status")), 
                       size = 1,  
                       prob = sigmoid(-0.2 + 0.5*X1 + 0.5*X2 + L2_prev - 0.6*E_prev))]
    }
    
    quant_df[, E := qbinom(get(paste0("F_E", k, "|L1", k, "_Status")), 
                    size = 1, 
                    prob = sigmoid(-1.5 + 0.5*X1 + 0.5*X2 + 0.5*L1 + 0.5*L2))]
    
    if (k > 0) {
      # Distributions of L1s and L2s in the Statusivors
      for (j in 1:k) {
        if (j < k) {
          quant_df[, paste0("F_L2", k - j, "|L1", k - j, "_Status") :=
                     cCopula(as.matrix(cbind(
                       get(paste0("F_Event|L1", k - j, "_L2", k - j - 1, "_Status_prev")),
                       get(paste0("F_L2", k - j, "|L1", k - j, "_Status_prev"))
                     )), 
                     copula = copulaL2)[, 2]]
          
          quant_df[, paste0("F_L1", k - j, "|Status") :=
                     cCopula(as.matrix(cbind(
                       get(paste0("F_Event|L1L2", k - j - 1, "_Status_prev")),
                       get(paste0("F_L1", k - j, "|Status_prev"))
                     )), 
                     copula = copulaL1)[, 2]]
        } 
        else {
          quant_df[, paste0("F_L2", k - j, "|L1", k - j, "_Status") :=
                     cCopula(as.matrix(cbind(
                       get(paste0("F_Event|L1", k - j, "_Status_prev")),
                       get(paste0("F_L2", k - j, "|L1", k - j, "_Status_prev"))
                     )), 
                     copula = copulaL2)[, 2]]
          
          quant_df[, paste0("F_L1", k - j, "|Status") :=
                     cCopula(as.matrix(cbind(
                       F_Event_Status_prev,
                       get(paste0("F_L1", k - j, "|Status_prev"))
                     )), 
                     copula = copulaL1)[, 2]]
        }
      }
    }
    
    for (j in 0:k) {
      if (j < k) {
        quant_df[, paste0("F_Event|L1", k - j, "_L2", k - j - 1, "_Status") :=
                   cCopula(as.matrix(cbind(
                     get(paste0("F_L2", k - j, "|L1", k - j, "_Status")),
                     get(paste0("F_Event|L1L2", k - j, "_Status"))
                   )), 
                   copula = copulaL2, inverse = TRUE)[, 2]]
        
        quant_df[, paste0("F_Event|L1L2", k - j - 1, "_Status") :=
                   cCopula(as.matrix(cbind(
                     get(paste0("F_L1", k - j, "|Status")),
                     get(paste0("F_Event|L1", k - j, "_L2", k - j - 1, "_Status"))
                   )), 
                   copula = copulaL1, inverse = TRUE)[, 2]]
      } else {
        quant_df[, paste0("F_Event|L1", k - j, "_Status") :=
                   cCopula(as.matrix(cbind(
                     get(paste0("F_L2", k - j, "|L1", k - j, "_Status")),
                     get(paste0("F_Event|L1L2", k - j, "_Status"))
                   )), 
                   copula = copulaL2, inverse = TRUE)[, 2]]
        
        quant_df[, F_Event_Status :=
                   cCopula(as.matrix(cbind(
                     get(paste0("F_L1", k - j, "|Status")),
                     get(paste0("F_Event|L1", k - j, "_Status"))
                   )), 
                   copula = copulaL1, inverse = TRUE)[, 2]]
      }
    }
    
    # hazard
    quant_df[, lambda    := exp(-2) * exp(alpha*E + beta*X2 + gamma*X2*E)]
    quant_df[, Surv_Time := qexp(F_Event_Status, rate = lambda)]
    quant_df[, Event := 1 * (Surv_Time < 1)]
    
    sim_df[Status == 1, paste0("Confounder1_", k) := quant_df$L1]
    sim_df[Status == 1, paste0("Confounder2_", k) := quant_df$L2]
    sim_df[Status == 1, paste0("Treatment_", k) := quant_df$E]
    sim_df[Status == 1, paste0("Event_", k) := quant_df$Event]
    sim_df[Status == 1, Surv_Time := (quant_df$Surv_Time + k)]
    sim_df[           , Status := 1 * (get(paste0("Event_", k)) == 0)]
  }
  
  sim_long_df <- sim_df %>%
    pivot_longer(cols = starts_with("Confounder1") | 
                   starts_with("Confounder2") | 
                   starts_with("Treatment") | 
                   starts_with("Event"),,
                 names_to = c("Variable", "Obs_Time"),
                 names_pattern = "(.*)_(\\d+)",
                 values_to = "value") %>%
    mutate(Obs_Time = as.integer(Obs_Time)) %>%
    pivot_wider(names_from = Variable, values_from = value) %>%
    arrange(ID, Obs_Time)
  
  df_long_clean <- sim_long_df %>%
    filter(!is.na(Confounder1) & !is.na(Confounder2) & !is.na(Treatment) & !is.na(Event))
  
  df_long_clean =subset(df_long_clean, select = -Status)
  
  return(df_long_clean)
}

data_df <- simulate_causal_inference_data(alpha = -0.5,
                                          beta  = -0.5, 
                                          gamma =  0.3,
                                          rho1  = -0.5, 
                                          rho2  =  0.3,
                                          K     =  5  ,
                                          n     = 10
                                          ) 
View(data_df)





alpha_gamma_list <- list(c(-0.5, 0.3), c(0.2, 0.1))

rho1_rho2_list <- list(c(0.9, 0.7), c(-0.5, 0.4), c(-0.1, -0.2))

n_values <- c(10000)
reps <- 1:500


for (ag in alpha_gamma_list) {
  for (rr in rho1_rho2_list) {
    for (n_val in n_values) {
      for (rep_id in reps) {
        
        alpha <- ag[1]
        gamma <- ag[2]
        rho1  <- rr[1]
        rho2  <- rr[2]

        data_df <- simulate_causal_inference_data(
          alpha = alpha,
          beta  = -0.5,
          gamma = gamma,
          rho1  = rho1,
          rho2  = rho2,
          K     = 5, 
          n     = n_val
        )
        
        filename <- sprintf(
          "data_alpha_%.1f_gamma%.1f_rho1_%.1f_rho2_%.1f_n%d_rep%d.csv",
          alpha, gamma, rho1, rho2, n_val, rep_id
        )
        
        write.csv(data_df, filename, row.names = FALSE)
      }
    }
  }
}

