import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler


class AssetPricingDataset(Dataset):
    def __init__(self, macro_data, ff_data, target_data):
        self.X_macro = torch.tensor(macro_data).float()
        self.X_ff = torch.tensor(ff_data).float()
        self.Y_targets = torch.tensor(target_data).float()

    def __len__(self):
        return len(self.X_macro)

    def __getitem__(self, idx):
        return {
            'macro_X': self.X_macro[idx],   # [lookback, macro_dim]
            'ff_X': self.X_ff[idx],          # [ff_dim]
            'target_Y': self.Y_targets[idx]  # [num_assets]
        }


def _fetch_data(start, end):
    """Fetch and align FF5 factors, macro indicators, and FF25 portfolios."""
    import getFamaFrenchFactors
    import pandas_datareader.data as web

    # --- Fama-French 5 factors ---
    factors_df = getFamaFrenchFactors.famaFrench5Factor()
    factors_df[['Mkt-RF', 'SMB', 'HML', 'RF']] /= 100
    factors_df['date_ff_factors'] = pd.to_datetime(factors_df['date_ff_factors'])
    factors_df = factors_df.set_index('date_ff_factors')
    factors_df = factors_df[factors_df.index >= start]

    # --- Macro indicators from FRED ---
    macro_tickers = {
        'Term_Spread': 'T10YFFM',
        'Default_Spread': 'AAAFFM',
        'Ind_Production': 'INDPRO',
        'Unemployment': 'UNRATE',
        'Consumer_Sentiment': 'UMCSENT',
    }
    macro_raw = web.DataReader(list(macro_tickers.values()), 'fred', start=start, end=end)
    macro_raw.columns = list(macro_tickers.keys())
    macro_raw = macro_raw.ffill().resample('ME').last()

    # --- FF 25 portfolios ---
    ff_portfolio = web.DataReader('25_Portfolios_5x5', 'famafrench', start=start, end=end)
    df_ret = ff_portfolio[0].replace([-99.99, -999], np.nan).dropna()
    df_ret = df_ret / 100
    df_ret.index = df_ret.index.to_timestamp(how='end').normalize()

    # --- Merge ---
    macro_raw.index = pd.to_datetime(macro_raw.index)
    factors_df.index = pd.to_datetime(factors_df.index)
    df_ret.index = pd.to_datetime(df_ret.index)

    combined = (
        factors_df
        .merge(macro_raw, left_index=True, right_index=True, how='inner')
        .merge(df_ret, left_index=True, right_index=True, how='inner')
        .dropna()
    )
    return combined


def _rolling_window(data, lookback):
    return np.array([data[i: i + lookback] for i in range(len(data) - lookback)])


def get_dataloaders(
    start='1964-01-01',
    end='2025-08-31',
    lookback=12,
    batch_size=64,
    train_ratio=0.8,
    val_ratio=0.1,
):
    """
    Fetches all data, builds rolling-window inputs, computes excess returns,
    fits StandardScaler on training split only, and returns DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, combined_val_test_loader, dims
    dims: dict with keys macro_dim, ff_dim, num_assets
    """
    combined = _fetch_data(start, end)

    ff_cols = combined.columns[:6]       # Mkt-RF, SMB, HML, RMW, CMA, RF
    macro_cols = combined.columns[6:11]  # 5 macro indicators
    port_cols = combined.columns[11:]    # 25 FF portfolios

    ff_arr = combined[ff_cols].values.astype(np.float32)
    macro_arr = combined[macro_cols].values.astype(np.float32)
    port_arr = combined[port_cols].values.astype(np.float32)
    rf_arr = combined['RF'].values.astype(np.float32)  # shape (T,)

    # Compute excess returns: R^e_{t,i} = R_{t,i} - RF_t
    excess_returns = port_arr - rf_arr[:, None]

    # Create rolling macro windows: window i spans rows [i, i+lookback)
    # X_macro[i] -> last obs at row i+lookback-1
    X_macro = _rolling_window(macro_arr, lookback)      # (N, lookback, macro_dim)
    # Align FF factors and excess returns with end of each window
    # X_ff[i]  = ff at row i+lookback-1  (contemporaneous with window end)
    # Y[i]     = excess_returns at row i+lookback  (one step ahead target)
    X_ff = ff_arr[lookback - 1: -1]                    # (N, ff_dim)
    Y = excess_returns[lookback:]                       # (N, num_assets)

    N = len(X_macro)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    # --- Scale features using training statistics only ---
    macro_scaler = StandardScaler()
    ff_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Reshape macro for scaler: (N*lookback, macro_dim) -> fit -> reshape back
    macro_train_flat = X_macro[:train_end].reshape(-1, X_macro.shape[2])
    macro_scaler.fit(macro_train_flat)

    ff_scaler.fit(X_ff[:train_end])
    y_scaler.fit(Y[:train_end])

    def scale(macro, ff, y):
        macro_flat = macro.reshape(-1, macro.shape[2])
        macro_s = macro_scaler.transform(macro_flat).reshape(macro.shape)
        ff_s = ff_scaler.transform(ff)
        y_s = y_scaler.transform(y)
        return macro_s, ff_s, y_s

    macro_s, ff_s, y_s = scale(X_macro, X_ff, Y)

    dataset = AssetPricingDataset(macro_s, ff_s, y_s)

    train_idx = range(0, train_end)
    val_idx = range(train_end, val_end)
    test_idx = range(val_end, N)
    combined_val_test_idx = range(train_end, N)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    combined_val_test_loader = DataLoader(Subset(dataset, combined_val_test_idx), batch_size=batch_size, shuffle=False)

    dims = {
        'macro_dim': X_macro.shape[2],
        'ff_dim': X_ff.shape[1],
        'num_assets': Y.shape[1],
    }

    return train_loader, val_loader, test_loader, combined_val_test_loader, dims
