import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def evaluate_GAN(generator, loader):
    """
    Evaluate a trained Generator on a DataLoader.

    Metrics
    -------
    sharpe_ratio            : annualised Sharpe of the SDF portfolio
    cross_sectional_r2      : R² from regressing mean realised returns on
                              mean model-implied weights (cross-sectional fit)
    mean_abs_pricing_error  : mean |E[M * R^e_i]| across assets
    max_pricing_error       : max  |E[M * R^e_i]| across assets
    mean_squared_pricing_error : mean (E[M * R^e_i])^2 across assets
    """
    generator.eval()
    all_sdf = []
    all_returns = []
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            macro = batch['macro_X']
            ff = batch['ff_X']
            returns = batch['target_Y']

            weights, _ = generator(macro, ff)
            sdf = 1 - (weights * returns).sum(dim=1)

            all_sdf.append(sdf.numpy())
            all_returns.append(returns.numpy())
            all_weights.append(weights.numpy())

    sdf = np.concatenate(all_sdf)           # (T,)
    returns = np.concatenate(all_returns)   # (T, num_assets)
    weights = np.concatenate(all_weights)   # (T, num_assets)

    results = {}

    # 1. Sharpe ratio of the SDF portfolio (annualised, monthly data)
    portfolio_returns = (weights * returns).sum(axis=1)
    results['sharpe_ratio'] = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12)

    # 2. Cross-sectional R²
    #    Regress mean realised excess returns on mean model-implied weights
    #    (weights serve as a proxy for factor loadings / risk prices)
    mean_returns = returns.mean(axis=0)           # (num_assets,)
    mean_weights = weights.mean(axis=0)           # (num_assets,)
    reg = LinearRegression().fit(mean_weights.reshape(-1, 1), mean_returns)
    pred_mean = reg.predict(mean_weights.reshape(-1, 1))
    ss_res = np.sum((mean_returns - pred_mean) ** 2)
    ss_tot = np.sum((mean_returns - mean_returns.mean()) ** 2)
    results['cross_sectional_r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # 3. Pricing errors: E[M_t * R^e_{t,i}] for each asset i
    pricing_errors = (sdf[:, None] * returns).mean(axis=0)   # (num_assets,)
    results['mean_abs_pricing_error'] = np.abs(pricing_errors).mean()
    results['max_pricing_error'] = np.abs(pricing_errors).max()
    results['mean_squared_pricing_error'] = (pricing_errors ** 2).mean()

    return results
