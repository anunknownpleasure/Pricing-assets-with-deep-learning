import torch
import torch.nn as nn
import torch.optim as optim
import numpy


class Generator(nn.Module):
    """
    Generator (SDF Network)

    Inputs:
    - macro_X: Macro time series [batch, seq_len, macro_dim]
    - ff_X: Fama-French factors [batch, ff_dim]

    Output:
    - omega: SDF portfolio weights [batch, num_assets]
    - h_t: Hidden macro states [batch, hidden_dim]
    """
    def __init__(self, macro_dim, ff_dim, hidden_dim, lstm_layers, num_assets):
        super().__init__()

        # LSTM for processing macro time series
        self.lstm = nn.LSTM(
            input_size=macro_dim,
            hidden_size=macro_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.05 if lstm_layers > 1 else 0
        )

        # Feedforward network: [hidden_states + FF_factors] -> portfolio_weights
        self.fc1 = nn.Linear(macro_dim + ff_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Output layer: portfolio weights for each asset
        self.fc3 = nn.Linear(hidden_dim//2, num_assets)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, macro_X, ff_X):
        # Process macro time series with LSTM
        _, (h_n, c_n) = self.lstm(macro_X)
        h_t = h_n[-1]  # Take last hidden state

        # Combine macro states with FF factors
        combined = torch.cat([h_t, ff_X], dim=1)

        # Feedforward layers
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Output: portfolio weights
        omega = self.fc3(x)

        return omega, h_t


class Discriminator(nn.Module):
    """
    Discriminator (Conditioning Network)

    Inputs:
    - h_t: Hidden macro states [batch, hidden_dim]
    - ff_X: FF factors [batch, ff_dim]

    Output:
    - g: Conditioning instruments [batch, num_instruments]
    """
    def __init__(self, hidden_dim, ff_dim, num_instruments, hidden_layer):
        super().__init__()

        input_dim = hidden_dim + ff_dim

        self.fc1 = nn.Linear(input_dim, hidden_layer)
        self.bn1 = nn.BatchNorm1d(hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer // 2)
        self.bn2 = nn.BatchNorm1d(hidden_layer // 2)
        self.fc3 = nn.Linear(hidden_layer // 2, num_instruments)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, h_t, ff_X):
        # Combine macro states with FF factors
        x = torch.cat([h_t, ff_X], dim=1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Output: conditioning instruments
        g = torch.tanh(self.fc3(x))  # Bounded in [-1, 1]

        return g
    
class GAN_loss:
    def __init__(self):
        # no instance state required; keep initializer to avoid indentation errors
        pass

    @staticmethod
    def calculate_SDF(weights, returns):

        weighted_returns = (weights * returns).sum(dim=1, keepdim=True)
        sdf_m = 1 - weighted_returns

        return sdf_m

    @staticmethod
    def no_arbitrage_loss(weights, returns, g_instruments, num_assets):

        sdf = GAN_loss.calculate_SDF(weights, returns)
        g_expanded = g_instruments.mean(dim=1, keepdim=True).expand(-1, num_assets)
        pricing_errors = sdf * returns * g_expanded
        mean_errors = pricing_errors.mean(dim=0)
        loss = (mean_errors ** 2).mean()
        return loss

    @staticmethod
    def discriminator_loss(weights, returns, g_instruments, num_assets, reg = True):

        na_loss = GAN_loss.no_arbitrage_loss(weights, returns, g_instruments, num_assets)
        D_regularization_term = 0.01 * (g_instruments ** 2).mean()

        if reg:
            return -na_loss + D_regularization_term
        else:
            return -na_loss

    @staticmethod
    def generator_loss(weights, returns, g_instruments, num_assets, reg = True):

        na_loss = GAN_loss.no_arbitrage_loss(weights, returns, g_instruments, num_assets)
        G_regularization_term = 0.01 * (weights ** 2).mean()

        if reg:
            return na_loss + G_regularization_term
        else:
            return na_loss
        

