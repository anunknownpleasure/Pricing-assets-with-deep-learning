from sklearn.linear_model import LinearRegression
import numpy as np


def fama_french_5(train_loader, test_loader):
    """
    Baseline 1: Fama-French 5-Factor Model

    Traditional factor model:
    R_i,t = α_i + β'·F_t + ε_i,t
    """

    # Collect all data
    train_returns = []
    train_ff = []
    test_returns = []
    test_ff = []

    for batch in train_loader:
        train_returns.append(batch['target_Y'].numpy())
        train_ff.append(batch['ff_X'].numpy())

    for batch in test_loader:
        test_returns.append(batch['target_Y'].numpy())
        test_ff.append(batch['ff_X'].numpy())

    train_returns = np.concatenate(train_returns, axis=0)
    train_ff = np.concatenate(train_ff, axis=0)
    test_returns = np.concatenate(test_returns, axis=0)
    test_ff = np.concatenate(test_ff, axis=0)

    # Use first 5 FF factors (excluding RF)
    train_ff = train_ff[:, :5]  # Mkt-RF, SMB, HML, RMW, CMA
    test_ff = test_ff[:, :5]

    # Estimate factor loadings for each portfolio
    betas = []
    alphas = []

    for i in range(train_returns.shape[1]):
        model = LinearRegression()
        model.fit(train_ff, train_returns[:, i])
        betas.append(model.coef_)
        alphas.append(model.intercept_)

    betas = np.array(betas)
    alphas = np.array(alphas)

    # Create SDF = mean-variance efficient combination of factors
    mean_factors = train_ff.mean(axis=0)
    cov_factors = np.cov(train_ff.T)

    try:
        weights = np.linalg.solve(cov_factors, mean_factors)
        train_sdf_returns = (train_ff @ weights).flatten()
        test_sdf_returns = (test_ff @ weights).flatten()
    except:
        # If singular, use equal weights
        train_sdf_returns = train_ff.mean(axis=1)
        test_sdf_returns = test_ff.mean(axis=1)

    # Compute metrics
    train_sharpe = (train_sdf_returns.mean() / train_sdf_returns.std()) * np.sqrt(12)
    test_sharpe = (test_sdf_returns.mean() / test_sdf_returns.std()) * np.sqrt(12)
    avg_alpha = np.abs(alphas).mean()

    return {
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'avg_alpha': avg_alpha,
        'train_returns': train_sdf_returns,
        'test_returns': test_sdf_returns
    }
    

def linear_mv(train_loader, test_loader):
    """
    Baseline 2: Simple Linear Mean-Variance Portfolio

    Maximize: ω'μ - λ·ω'Σω
    """

    # Collect returns
    train_returns = []
    test_returns = []

    for batch in train_loader:
        train_returns.append(batch['target_Y'].numpy())

    for batch in test_loader:
        test_returns.append(batch['target_Y'].numpy())

    train_returns = np.concatenate(train_returns, axis=0)
    test_returns = np.concatenate(test_returns, axis=0)

    # Estimate mean and covariance
    mean_ret = train_returns.mean(axis=0)
    cov_mat = np.cov(train_returns.T)

    # Solve for tangency portfolio
    try:
        inv_cov = np.linalg.inv(cov_mat)
        weights = inv_cov @ mean_ret
        weights = weights / np.abs(weights).sum()
    except:
        weights = np.ones(train_returns.shape[1]) / train_returns.shape[1]

    # Portfolio returns
    train_portfolio_ret = (train_returns @ weights).flatten()
    test_portfolio_ret = (test_returns @ weights).flatten()

    # Compute Sharpe ratios
    train_sharpe = (train_portfolio_ret.mean() / train_portfolio_ret.std()) * np.sqrt(12)
    test_sharpe = (test_portfolio_ret.mean() / test_portfolio_ret.std()) * np.sqrt(12)


    return {
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'weights': weights,
        'train_returns': train_portfolio_ret,
        'test_returns': test_portfolio_ret
    }
