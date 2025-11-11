import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from getFamaFrenchFactors import famaFrench5Factor
from sklearn.preprocessing import StandardScaler

# --- 1. Helper Function: Rolling Window ---

def rolling_window(data, lookback):
  """
  Creates rolling windows from time-series data.
  """
  x_rolling = []
  for i in range(len(data) - lookback):
    x_rolling.append(data[i: i + lookback])
  return np.array(x_rolling)

# --- 2. PyTorch Dataset Class ---

class AssetPricingDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the aligned
    Macro, Fama-French, and Portfolio data.
    """
    def __init__(self, macro_data, ff_data, target_data):
        self.X_macro = torch.tensor(macro_data).float()
        self.X_ff = torch.tensor(ff_data).float()
        self.Y_targets = torch.tensor(target_data).float()

    def __len__(self):
        return len(self.X_macro)

    def __getitem__(self, idx):
        return {
            'macro_X': self.X_macro[idx],    # Shape: [lookback, num_macro_features]
            'ff_X': self.X_ff[idx],          # Shape: [num_ff_features]
            'target_Y': self.Y_targets[idx]  # Shape: [num_portfolios]
        }

# --- 3. Main Data Fetching & Processing Function ---

def get_dataloaders(
    lookback=12,
    batch_size=64,
    train_pct=0.7,
    val_pct=0.15,
    start_date='1964-01-01',
    end_date='2025-08-31'
):
    """
    Fetches, processes, aligns, and splits all data, returning
    PyTorch DataLoaders for training, validation, and testing.
    """
    
    # --- 3a. Fetch Fama-French 5 Factors ---
    factors_df = famaFrench5Factor()
    factors_df[['Mkt-RF', 'SMB', 'HML', 'RF']] = factors_df[['Mkt-RF', 'SMB', 'HML', 'RF']] / 100
    FFdata = factors_df.iloc[6:] # Start from 1964-01-31
    FFdata['date_ff_factors'] = pd.to_datetime(FFdata['date_ff_factors'])
    FFdata = FFdata.set_index('date_ff_factors')

    # --- 3b. Fetch Macroeconomic Data ---
    long_macro_tickers = {
        'Term_Spread': 'T10YFFM',
        'Default_Spread': 'AAAFFM',
        'Ind_Production': 'INDPRO',
        'Unemployment': 'UNRATE',
        'Consumer_Sentiment': 'UMCSENT'
    }
    macro_data = web.DataReader(
        list(long_macro_tickers.values()), 'fred', start=start_date, end=end_date
    )
    macro_data.columns = list(long_macro_tickers.keys())
    macro_data = macro_data.resample('ME').last().ffill()
    macro_data.index = pd.to_datetime(macro_data.index)

    # --- 3c. Fetch Fama-French 25 Portfolios ---
    ff_portfolio = web.DataReader('25_Portfolios_5x5', 'famafrench', start=start_date, end=end_date)
    df_returns_25 = ff_portfolio[0]
    df_returns_25 = df_returns_25.replace([-99.99, -999], np.nan)
    df_returns_25 = (df_returns_25 / 100)
    df_returns_25.index = df_returns_25.index.to_timestamp(how='end').date
    df_returns_25.index = pd.to_datetime(df_returns_25.index)

    # --- 3d. Merge and Clean Data ---
    combined_data_FF_macro = pd.merge(FFdata, macro_data, left_index=True, right_index=True, how='inner')
    combined_data = pd.merge(combined_data_FF_macro, df_returns_25, left_index=True, right_index=True, how='inner')
    
    # CRITICAL FIX: Forward-fill to prevent losing rows, then drop remaining NaNs
    combined_data = combined_data.ffill().dropna()

    # --- 3e. CRITICAL FIX: Calculate Excess Returns ---
    # This was your Cell 14. The model must predict *excess* returns.
    portfolio_columns = df_returns_25.columns
    rf = combined_data['RF']
    portfolio_excess_returns = combined_data[portfolio_columns].subtract(rf, axis=0)
    
    # Replace raw returns with excess returns in the main DataFrame
    combined_data[portfolio_columns] = portfolio_excess_returns
    
    # Define features and targets
    FF_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    macro_columns = list(long_macro_tickers.keys())
    # portfolio_columns is already defined

    no_of_FF_features = len(FF_columns)
    no_macro_features = len(macro_columns)
    no_of_portfolios = len(portfolio_columns)

    # --- 3f. Scale Data ---
    # We scale features and targets separately
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()

    # Fit scalers *only* on the training data portion to prevent leakage
    n_total = len(combined_data)
    n_train = int(n_total * train_pct)
    
    feature_data = combined_data[FF_columns + macro_columns]
    target_data = combined_data[portfolio_columns]

    # Fit on train data
    scaler_features.fit(feature_data.iloc[:n_train])
    scaler_targets.fit(target_data.iloc[:n_train])
    
    # Transform all data
    processed_features = scaler_features.transform(feature_data)
    processed_targets = scaler_targets.transform(target_data)

    # --- 3g. Slicing and Alignment (CRITICAL FIXES APPLIED) ---
    ff_data = processed_features[:, :no_of_FF_features]
    macro_data = processed_features[:, no_of_FF_features:]
    portfolio_data_scaled = processed_targets # These are now scaled excess returns

    # Create rolling window on macro data
    X_macro_rolled = rolling_window(macro_data, lookback) 

    # Align X(t-1) with Y(t)
    X_ff_aligned = ff_data[lookback-1:-1]
    Y_targets_aligned = portfolio_data_scaled[lookback:]
    
    # Trim macro_rolled to match the new shorter lengths
    X_macro_rolled = X_macro_rolled[:-1]

    print(f"--- Data Shapes after Alignment ---")
    print(f"X_macro_rolled shape: {X_macro_rolled.shape}")
    print(f"X_ff_aligned shape:   {X_ff_aligned.shape}")
    print(f"Y_targets_aligned shape: {Y_targets_aligned.shape}")

    # --- 3h. Create Datasets and DataLoaders (CRITICAL FIX APPLIED) ---
    data = AssetPricingDataset(X_macro_rolled, X_ff_aligned, Y_targets_aligned)
    
    n_samples = len(data)
    train_idx = int(n_samples * train_pct)
    val_idx = int(n_samples * (train_pct + val_pct))

    indices = list(range(n_samples))
    
    # Use Subset for chronological split
    train_data = Subset(data, indices[:train_idx])
    val_data = Subset(data, indices[train_idx:val_idx])
    test_data = Subset(data, indices[val_idx:])

    print(f"Total samples: {n_samples}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, no_macro_features, no_of_FF_features, no_of_portfolios

# --- 4. Test block (to run this file directly) ---
if __name__ == '__main__':
    print("Testing data_loader.py...")
    
    train_loader, val_loader, test_loader, macro_dim, ff_dim, port_dim = get_dataloaders(
        lookback=12,
        batch_size=64,
        train_pct=0.7,
        val_pct=0.15
    )
    
    print(f"\nDimensions: Macro={macro_dim}, FF={ff_dim}, Portfolios={port_dim}")
    
    # Check one batch
    print("\n--- Checking one batch from train_loader ---")
    batch = next(iter(train_loader))
    print(f"macro_X batch shape: {batch['macro_X'].shape}")
    print(f"ff_X batch shape:    {batch['ff_X'].shape}")
    print(f"target_Y batch shape: {batch['target_Y'].shape}")
    
    print("\nData loader test complete.")