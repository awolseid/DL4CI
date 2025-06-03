import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader, TensorDataset

from scripts.training_and_testing import train_denom_model, train_numer_model


seed = 42
def cross_validate_denom_model(model_class, 
                               dataset,
                               static_dim, 
                               longit_dim, 
                               hidden_dim, 
                               epochs,
                               collate_fn,
                               lr, 
                               batch_size, 
                               n_splits):
    
    # Extract labels for stratification
    labels = np.array([sample["labels"][-1] for sample in dataset])

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_val_log_losses = []
    all_summaries      = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, 
                                  batch_size    = batch_size, 
                                  shuffle       = True, 
                                  collate_fn    = collate_fn)
        val_loader = DataLoader(val_subset, 
                                batch_size      = batch_size, 
                                shuffle         = False, 
                                collate_fn      = collate_fn)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            
        model     = model_class(longit_dim, static_dim, hidden_dim).to(device)
        summaries = train_denom_model(model         = model, 
                                      train_dataset = train_subset, 
                                      val_dataset   = val_subset, 
                                      batch_size    = batch_size, 
                                      lr            = lr, 
                                      epochs        = epochs)

        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for x_static, x_longit, y in val_loader:
                x_static, x_longit = x_static.to(device), x_longit.to(device)
                y_pred  = model(x_static, x_longit)
                probs   = torch.sigmoid(y_pred).cpu().numpy().flatten()
                targets = y.cpu().numpy().flatten()

                mask = targets  != -1
                filtered_probs   = probs[mask]
                filtered_targets = targets[mask]

                all_probs.extend(filtered_probs)
                all_targets.extend(filtered_targets)

        logloss = log_loss(all_targets, all_probs)
        print(f"Log Loss (Fold {fold + 1}): {logloss:.4f}")
        all_val_log_losses.append(logloss)
        all_summaries.append(summaries)

    avg_log_loss = np.mean(all_val_log_losses)
    print(f"\nAverage Log Loss across {n_splits} folds: {avg_log_loss:.4f}")
    return all_summaries, all_val_log_losses

# def grid_search_denom_model(model_class, 
#                             dataset,
#                             param_grid, 
#                             epochs, 
#                             collate_fn,
#                             n_splits=5
#                            ):
#     grid                 = ParameterGrid(param_grid)
#     static_dim           = len(dataset[0]["static"]) 
#     longit_dim           = dataset[0]["longitudinal"].shape[1]
#     best_params          = None
#     best_log_loss        = float('inf')
#     best_model_summaries = None
    
#     for params in grid:
#         print(f"\n=== Training with Hyperparameters: {params} ===")
        
#         hidden_dim = params['hidden_dim']
#         lr         = params['lr']
#         batch_size = params['batch_size']
        
#         summaries, log_losses = cross_validate_denom_model(model_class = model_class, 
#                                                            dataset     = dataset, 
#                                                            static_dim  = static_dim, 
#                                                            longit_dim  = longit_dim,
#                                                            hidden_dim  = hidden_dim, 
#                                                            epochs      = epochs, 
#                                                            lr          = lr, 
#                                                            batch_size  = batch_size, 
#                                                            n_splits    = n_splits,
#                                                            collate_fn  = collate_fn
#                                                            )
        
#         avg_log_loss = np.mean(log_losses)
#         print(f"Average Log Loss for {params}: {avg_log_loss:.4f}")
        
#         if avg_log_loss < best_log_loss:
#             best_log_loss        = avg_log_loss
#             best_params          = params
#             best_model_summaries = summaries
    
#     print(f"\nBest Hyperparameters: {best_params}")
#     print(f"     Lowest Validation Log Loss: {best_log_loss:.10f}")
    
#     return best_params, best_model_summaries, best_log_loss





# def grid_search_cv_numer_model(model_class : nn.Module, 
#                                dataset     : TensorDataset, 
#                                param_grid  : dict,
#                                epochs      : int           = 10, 
#                                n_splits    : int           = 5
#                               ):

#     device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     grid                 = ParameterGrid(param_grid)
#     best_params          = None
#     best_val_loss        = float('inf')
#     best_model_summaries = None

#     labels = dataset.tensors[1].squeeze().numpy()
#     skf    = StratifiedKFold(n_splits     = n_splits, 
#                              shuffle      = True, 
#                              random_state =seed)

#     for params in grid:
#         print(f"\n=== Testing hyperparameters: {params} ===")
#         val_losses_all_folds = []

#         for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
#             print(f"  Fold {fold + 1}/{n_splits}")

#             train_subset = Subset(dataset, train_idx)
#             val_subset   = Subset(dataset, val_idx)

#             torch.manual_seed(42)
#             if device == "cuda":
#                 torch.cuda.manual_seed(42)
#                 torch.cuda.manual_seed_all(seed)
            
#             model = model_class(input_dim  = dataset.tensors[0].shape[1], 
#                                 hidden_dim = params['hidden_dim'])

#             summaries = train_numer_model(model         = model,
#                                           train_dataset = train_subset,
#                                           val_dataset   = val_subset,
#                                           batch_size    = params['batch_size'],
#                                           epochs        = epochs,
#                                           lr            = params['lr']
#                                         )

#             if summaries["val_losses"]:
#                 fold_val_loss = summaries["val_losses"][-1]
#                 val_losses_all_folds.append(fold_val_loss)

#         if val_losses_all_folds:
#             avg_val_loss = np.mean(val_losses_all_folds)
#             print(f"  ➤ Average Log Loss (val) = {avg_val_loss:.4f}")

#             if avg_val_loss < best_val_loss:
#                 best_val_loss        = avg_val_loss
#                 best_params          = params
#                 best_model_summaries = summaries

#     print(f"\n✅ Best Hyperparameters: {best_params}")
#     print(f"   Lowest Validation Log Loss: {best_val_loss:.10f}")
#     return best_params, best_model_summaries, best_val_loss



def grid_search_denom_model(model_class, 
                            dataset,
                            param_grid, 
                            collate_fn,
                            n_splits=5):
    grid                 = ParameterGrid(param_grid)
    static_dim           = len(dataset[0]["static"]) 
    longit_dim           = dataset[0]["longitudinal"].shape[1]
    best_params          = None
    best_log_loss        = float('inf')
    best_model_summaries = None
    
    for params in grid:
        print(f"\n=== Training with Hyperparameters: {params} ===")
        
        hidden_dim = params['hidden_dim']
        lr         = params['lr']
        batch_size = params['batch_size']
        epochs     = params['epochs']

        summaries, log_losses = cross_validate_denom_model(
            model_class = model_class, 
            dataset     = dataset, 
            static_dim  = static_dim, 
            longit_dim  = longit_dim,
            hidden_dim  = hidden_dim, 
            epochs      = epochs, 
            lr          = lr, 
            batch_size  = batch_size, 
            n_splits    = n_splits,
            collate_fn  = collate_fn
        )
        
        avg_log_loss = np.mean(log_losses)
        print(f"Average Log Loss for {params}: {avg_log_loss:.4f}")
        
        if avg_log_loss < best_log_loss:
            best_log_loss        = avg_log_loss
            best_params          = params
            best_model_summaries = summaries
    
    print(f"\n✅ Best Hyperparameters: {best_params}")
    print(f"   Lowest Validation Log Loss: {best_log_loss:.10f}")
    
    return best_params, best_model_summaries, best_log_loss



def grid_search_cv_numer_model(model_class, 
                               dataset, 
                               param_grid, 
                               n_splits=5):

    device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid                 = ParameterGrid(param_grid)
    best_params          = None
    best_val_loss        = float('inf')
    best_model_summaries = None

    labels = dataset.tensors[1].squeeze().numpy()
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for params in grid:
        print(f"\n=== Testing hyperparameters: {params} ===")
        val_losses_all_folds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
            print(f"  Fold {fold + 1}/{n_splits}")

            train_subset = Subset(dataset, train_idx)
            val_subset   = Subset(dataset, val_idx)

            torch.manual_seed(42)
            if device == "cuda":
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            
            model = model_class(
                input_dim  = dataset.tensors[0].shape[1], 
                hidden_dim = params['hidden_dim']
            )

            summaries = train_numer_model(
                model         = model,
                train_dataset = train_subset,
                val_dataset   = val_subset,
                batch_size    = params['batch_size'],
                epochs        = params['epochs'],  # ✅ now from grid
                lr            = params['lr']
            )

            if summaries["val_losses"]:
                fold_val_loss = summaries["val_losses"][-1]
                val_losses_all_folds.append(fold_val_loss)

        if val_losses_all_folds:
            avg_val_loss = np.mean(val_losses_all_folds)
            print(f"  ➤ Average Log Loss (val) = {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss        = avg_val_loss
                best_params          = params
                best_model_summaries = summaries

    print(f"\n✅ Best Hyperparameters: {best_params}")
    print(f"   Lowest Validation Log Loss: {best_val_loss:.10f}")
    
    return best_params, best_model_summaries, best_val_loss
