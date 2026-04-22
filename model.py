import torch
import torch.nn as nn


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
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.05 if lstm_layers > 1 else 0
        )

        # Feedforward network: [hidden_states + FF_factors] -> portfolio_weights
        self.fc1 = nn.Linear(hidden_dim + ff_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Output layer: portfolio weights for each asset
        self.fc3 = nn.Linear(hidden_dim // 2, num_assets)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, macro_X, ff_X):
        # Process macro time series with LSTM
        _, (h_n, c_n) = self.lstm(macro_X)
        h_t = h_n[-1]  # Take last hidden state [batch, hidden_dim]

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

        # Output: conditioning instruments, bounded in [-1, 1]
        g = torch.tanh(self.fc3(x))

        return g


class GAN_loss:
    def __init__(self):
        pass

    @staticmethod
    def calculate_SDF(weights, returns):
        weighted_returns = (weights * returns).sum(dim=1, keepdim=True)
        sdf_m = 1 - weighted_returns
        return sdf_m

    @staticmethod
    def no_arbitrage_loss(weights, returns, g_instruments):
        """
        Computes mean-squared pricing errors for all (asset, instrument) pairs.

        Moment condition: E[M_t * R^e_{t,i} * g_{t,j}] = 0  for all i, j.
        Loss = mean over (i,j) of ( mean_t[M_t * R^e_{t,i} * g_{t,j}] )^2
        """
        sdf = GAN_loss.calculate_SDF(weights, returns)   # [batch, 1]
        sdf_R = sdf * returns                             # [batch, num_assets]
        batch_size = sdf_R.shape[0]

        # moment_matrix[i, j] = mean_t( M_t * R_{t,i} * g_{t,j} )
        moment_matrix = torch.einsum('ba,bj->aj', sdf_R, g_instruments) / batch_size
        loss = (moment_matrix ** 2).mean()
        return loss

    @staticmethod
    def discriminator_loss(weights, returns, g_instruments, reg=True):
        na_loss = GAN_loss.no_arbitrage_loss(weights, returns, g_instruments)
        if reg:
            return -na_loss + 0.01 * (g_instruments ** 2).mean()
        return -na_loss

    @staticmethod
    def generator_loss(weights, returns, g_instruments, reg=True):
        na_loss = GAN_loss.no_arbitrage_loss(weights, returns, g_instruments)
        if reg:
            return na_loss + 0.01 * (weights ** 2).mean()
        return na_loss
