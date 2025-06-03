import torch
import torch.nn as nn


# class LSTM4TreatmentExposureDenominatorModel(nn.Module):
#     def __init__(self, longit_input_dim, static_input_dim, hidden_dim):
#         super(LSTM4TreatmentExposureDenominatorModel, self).__init__()
#         self.lstm       = nn.LSTM(longit_input_dim, hidden_dim, batch_first=True)
#         self.fc_static  = nn.Linear(static_input_dim, hidden_dim)
#         self.classifier = nn.Linear(hidden_dim * 2, 1)

#     def forward(self, x_static, x_longit):
#         lstm_out, _ = self.lstm(x_longit)
#         static_proj = self.fc_static(x_static).unsqueeze(1).expand(-1, lstm_out.size(1), -1)
#         concat      = torch.cat([lstm_out, static_proj], dim=-1)
#         logits      = self.classifier(concat).squeeze(-1)
#         return logits


class LSTM4TreatmentExposureDenominatorModel(nn.Module):
    def __init__(self, longit_input_dim, static_input_dim, hidden_dim, dropout_prob=0.3):
        super(LSTM4TreatmentExposureDenominatorModel, self).__init__()

        self.lstm = nn.LSTM(longit_input_dim, hidden_dim, batch_first=True)

        # Static path
        self.fc_static = nn.Linear(static_input_dim, hidden_dim)
        self.static_bn = nn.BatchNorm1d(hidden_dim)  # normalize over batch

        # Optional: normalize LSTM output (apply after reshaping)
        self.lstm_bn = nn.BatchNorm1d(hidden_dim)  # optional, comment out if not used

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_static, x_longit):
        # LSTM path
        lstm_out, _ = self.lstm(x_longit)  # [B, T, H]
        
        # Optional batchnorm over LSTM output
        B, T, H = lstm_out.size()
        lstm_out = self.lstm_bn(lstm_out.contiguous().view(-1, H))  # [B*T, H]
        lstm_out = lstm_out.view(B, T, H)  # back to [B, T, H]

        # Static path
        static_proj = self.fc_static(x_static)       # [B, H]
        static_proj = self.static_bn(static_proj)    # batch norm over batch
        static_proj = self.dropout(static_proj)      # dropout
        static_proj = static_proj.unsqueeze(1).expand(-1, T, -1)  # [B, T, H]

        # Combine and classify
        concat = torch.cat([lstm_out, static_proj], dim=-1)  # [B, T, 2H]
        concat = self.dropout(concat)
        logits = self.classifier(concat).squeeze(-1)          # [B, T]
        return logits

    


class TreatmentExposureNumeratorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(TreatmentExposureNumeratorModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        return self.network(x)

