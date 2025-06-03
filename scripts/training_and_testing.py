import pandas as pd

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.helper_funs import collate_fn


def train_denom_model(model         : nn.Module, 
                      train_dataset : pd.DataFrame, 
                      batch_size    : int, 
                      epochs        : int, 
                      lr            : float,
                      val_dataset   : pd.DataFrame  = None 
                      ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None
    
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    summaries = {"train_losses": [], "val_losses": [],
                 "train_accuracies": [], "val_accuracies": []
                }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for x_static, x_longit, y in train_loader:
            x_static, x_longit, y = x_static.to(device), x_longit.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(x_static, x_longit)
            mask   = (y != -1)
            loss   = criterion(y_pred, y)
            loss   = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y_prob   = torch.sigmoid(y_pred)
            y_label  = (y_prob >= 0.5).float()
            correct += ((y_label == y) * mask).sum().item()
            total   += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total

        summaries["train_losses"].append(avg_train_loss)
        summaries["train_accuracies"].append(train_accuracy)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0

            with torch.inference_mode():
                for x_static, x_longit, y in val_loader:
                    x_static, x_longit, y = x_static.to(device), x_longit.to(device), y.to(device)

                    y_pred    = model(x_static, x_longit)
                    mask      = (y != -1)
                    loss      = criterion(y_pred, y)
                    loss      = (loss * mask).sum() / mask.sum()
                    val_loss += loss.item()

                    y_prob   = torch.sigmoid(y_pred)
                    y_label  = (y_prob >= 0.5).float()
                    correct += ((y_label == y) * mask).sum().item()
                    total   += mask.sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total

            summaries["val_losses"].append(avg_val_loss)
            summaries["val_accuracies"].append(val_accuracy)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 or (epoch + 1) == epochs:
                print(f"Epoch {epoch + 1}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss   = {avg_val_loss:.4f} | "
                      f"Train Acc  = {train_accuracy:.4f}, "
                      f"Val Acc    = {val_accuracy:.4f}")
        else:
            if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 or (epoch + 1) == epochs:
                print(f"Epoch {epoch + 1}: "
                      f"Train: Loss = {avg_train_loss:.4f}, "
                      f"Acc = {train_accuracy:.4f}")

    return summaries



class GetDenomModelPredictions:
  def __init__(self):
    self.pred_probs  = None
    self.pred_labels = None

  def test_denom_model(self, trained_model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()

    all_pred_probs, all_true_labels = [], []

    with torch.inference_mode():
        for x_static, x_longit, y in data_loader:
            x_static = x_static.to(device)
            x_longit = x_longit.to(device)
            y = y.to(device)

            logits = trained_model(x_static, x_longit)
            pred_probs = torch.sigmoid(logits)

            mask = (y != -1)

            all_pred_probs.append(pred_probs[mask].cpu())
            all_true_labels.append(y[mask].cpu())

    self.pred_probs = torch.cat(all_pred_probs).numpy()
    self.pred_labels = (self.pred_probs >= 0.5).astype(int)

    flat_true_labels = torch.cat(all_true_labels).int().numpy()
    print(f"Denom Model Test accuracy: {accuracy_score(flat_true_labels, self.pred_labels)}.")









# def train_numer_model(model         : nn.Module,
#                     train_dataset : pd.DataFrame,
#                     val_dataset   : pd.DataFrame = None, 
#                     batch_size    : int          = 32, 
#                     epochs        : int          = 50, 
#                     lr            : float        = 0.001):
  
#   train_loader = DataLoader(dataset    = train_dataset, 
#                             batch_size = batch_size, 
#                             shuffle    = True)
#   if val_dataset is not None:
#      val_loader = DataLoader(dataset    = val_dataset, 
#                              batch_size = batch_size)
#   else:
#      val_loader = None
    
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   model = model.to(device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#   criterion = nn.BCELoss()

#   summaries = {"train_losses": [], "val_losses": [], "train_accuracies": [], "val_accuracies": []}

#   for epoch in range(epochs):
#       model.train()
#       train_loss = 0.0
#       correct, total = 0, 0

#       for X, y in train_loader:
#           X, y = X.to(device), y.to(device)
#           optimizer.zero_grad()
#           y_pred = model(X)
#           loss   = criterion(y_pred, y)
#           loss.backward()
#           optimizer.step()

#           train_loss += loss.item()
#           preds       = (y_pred >= 0.5).float()
#           correct    += (preds == y).sum().item()
#           total      += y.size(0)

#       avg_train_loss = train_loss / len(train_loader)
#       train_accuracy = correct / total
#       summaries["train_losses"].append(avg_train_loss)
#       summaries["train_accuracies"].append(train_accuracy)

#       if val_loader:
#           model.eval()
#           val_loss = 0.0
#           correct, total = 0, 0

#           with torch.no_grad():
#               for X, y in val_loader:
#                   X, y     = X.to(device), y.to(device)
#                   y_pred   = model(X)
#                   loss     = criterion(y_pred, y.view(-1, 1))

#                   val_loss += loss.item()
#                   preds     = (y_pred >= 0.5).float()
#                   correct  += (preds == y).sum().item()
#                   total    += y.size(0)

#           avg_val_loss = val_loss / len(val_loader)
#           val_accuracy = correct / total
#           summaries["val_losses"].append(avg_val_loss)
#           summaries["val_accuracies"].append(val_accuracy)

#           if (epoch + 1) % 5 == 0 or (epoch + 1) == 1 or (epoch + 1) == epochs:
#               print(f"Epoch {epoch+1}: "
#                     f"Train Loss = {avg_train_loss:.4f}, "
#                     f"Val Loss   = {avg_val_loss:.4f} | "
#                     f"Train Acc  = {train_accuracy:.4f}, "
#                     f"Val Acc    = {val_accuracy:.4f}")
#       else:
#           if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 or (epoch + 1) == epochs:
#               print(f"Epoch {epoch+1}: "
#                     f"Train Loss = {avg_train_loss:.4f}, "
#                     f"Train Acc  = {train_accuracy:.4f}")

#   return summaries


def train_numer_model(model         : nn.Module,
                      train_dataset : pd.DataFrame,
                      batch_size    : int, 
                      epochs        : int, 
                      lr            : float,
                      val_dataset   : pd.DataFrame = None
                     ):
  
    train_loader = DataLoader(dataset    = train_dataset, 
                              batch_size = batch_size, 
                              shuffle    = True)
    if val_dataset is not None:
        val_loader = DataLoader(dataset    = val_dataset, 
                                batch_size = batch_size)
    else:
        val_loader = None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    summaries = {"train_losses": [], "val_losses": [], "train_accuracies": [], "val_accuracies": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss   = criterion(y_pred, y.view(-1,1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(y_pred) >= 0.5).float()
            y = y.view_as(preds)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total

        summaries["train_losses"].append(avg_train_loss)
        summaries["train_accuracies"].append(train_accuracy)

        if val_loader:
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    loss = criterion(y_pred, y.view(-1, 1))

                    val_loss += loss.item()
                    preds = (torch.sigmoid(y_pred) >= 0.5).float()
                    y = y.view_as(preds)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total

            summaries["val_losses"].append(avg_val_loss)
            summaries["val_accuracies"].append(val_accuracy)

            if (epoch + 1) % 5 == 0 or (epoch + 1) == 1 or (epoch + 1) == epochs:
                print(f"Epoch {epoch+1}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss   = {avg_val_loss:.4f} | "
                      f"Train Acc  = {train_accuracy:.4f}, "
                      f"Val Acc    = {val_accuracy:.4f}")
        else:
            if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 or (epoch + 1) == epochs:
                print(f"Epoch {epoch+1}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Train Acc  = {train_accuracy:.4f}")

    return summaries


def test_numer_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    criterion  = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    correct    = 0
    total      = 0

    all_probs  = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss   = criterion(y_pred, y.view(-1, 1))
            
            total_loss += loss.item()

            y_prob  = torch.sigmoid(y_pred)
            y_label = (y_prob >= 0.5).float()
            y = y.view_as(y_label)

            correct += (y_label == y).sum().item()
            total   += y.size(0)

            all_probs.append(y_prob.view(-1).cpu())
            all_preds.append(y_label.view(-1).cpu())
            all_labels.append(y.view(-1).cpu())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    y_probs = torch.cat(all_probs).numpy().flatten()
    y_preds = torch.cat(all_preds).numpy().flatten()
    y_true  = torch.cat(all_labels).numpy().flatten()

    print(f"Numer Model Test Accuracy: {accuracy:.4f}")

    return y_probs, y_preds, y_true


