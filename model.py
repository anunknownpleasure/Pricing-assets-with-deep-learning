
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
  def __init__(self, macro_dim, hidden_dim, lstm_layers, sdf_dim):
    super().__init__()
    # Define the LSTM layer for processing macro data
    self.lstm = nn.LSTM(input_size= macro_dim,
                        hidden_size=hidden_dim,
                        num_layers=lstm_layers,
                        batch_first=True,
                        dropout=0.2 if lstm_layers > 1 else 0)

    self.fc1 = nn.Linear(hidden_dim, hidden_dim)
    self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
    self.activation = nn.LeakyReLU(0.2)     # Better activation
    self.dropout = nn.Dropout(0.2)          # Dropout for regularization
    self.fc2 = nn.Linear(hidden_dim, sdf_dim)

  def forward(self, macro_X, ff_X):
    _, (h_n, c_n) = self.lstm(macro_X)
    h_t = h_n[-1]

    x = self.fc1(h_t)
    x = self.bn1(x)
    x = self.activation(x)
    x = self.dropout(x)
    f_t = self.fc2(x)

    return f_t, h_t


class Discriminator(nn.Module):

    def __init__(self, hidden_dim, num_assets, hidden_layer):
        # hidden_dim is the size of h_t (e.g., 8)
        # num_assets is the size of the output portfolio weights g_t (25)
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_layer)
        self.bn1 = nn.BatchNorm1d(hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer // 2)
        self.bn2 = nn.BatchNorm1d(hidden_layer // 2)
        self.fc3 = nn.Linear(hidden_layer // 2, num_assets)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, h_t):

        x = self.fc1(h_t)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        g_t = self.fc3(x)
        return g_t
