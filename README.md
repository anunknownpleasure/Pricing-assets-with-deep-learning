# Pricing-assets-with-deep-learning

This project explores the application of a Generative Adversarial Network (GAN) for estimating the Stochastic Discount Factor (SDF) in asset pricing, inspired by [Chen, Pelger & Zhu (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3350138). The core objective is to learn an SDF that satisfies the no-arbitrage condition, $E[M_t R^e_{t,i}] = 0$, where $M_t$ is the SDF and $R^e_{t,i}$ are excess asset returns.

The approach uses a GAN consisting of a **Generator** (SDF network) and a **Discriminator** (conditioning network). The Generator passes a 12-month rolling window of 15 FRED macroeconomic indicators through an LSTM, concatenates the final hidden state with contemporaneous Fama-French 6 factors, and outputs portfolio weights $\omega_t$; the SDF is then $M_t = 1 - \omega_t^\top R^e_t$. The Discriminator has its own separate LSTM over the same macro window, whose hidden state is concatenated with FF6 factors and fed through a feedforward network to produce conditioning instruments $g_t$. The adversarial loss enforces the no-arbitrage moment conditions $E[M_t R^e_{t,i} g_{t,j}] = 0$ over all $(i, j)$ pairs simultaneously.

Training stability is improved via **Generator pre-training** (50 epochs of Sharpe-ratio maximisation before the adversarial phase) and **ExponentialLR decay** ($\gamma = 0.995$) applied to both optimisers throughout adversarial training.

The test assets are the 25 Fama-French portfolios sorted by size and book-to-market. Hyperparameters are selected by Bayesian optimisation (Optuna, 20 trials) maximising validation Sharpe ratio. To reduce variance from random initialisation, metrics are aggregated over an ensemble of independently trained models.

## Module Layout

| File | Purpose |
|------|---------|
| `data_loader.py` | Data fetching, excess returns, rolling windows, train-only scaler, DataLoaders |
| `model.py` | `Generator`, `Discriminator`, `GAN_loss` |
| `train.py` | `pretrain_generator`, `training_fn` — pre-training and adversarial loop |
| `evaluate.py` | `evaluate_GAN` — Sharpe, cross-sectional R², pricing errors |
| `ensemble.py` | `run_ensemble`, `print_ensemble_summary` |
| `tune.py` | `run_optuna_study` — Bayesian hyperparameter search |
| `baseline_models.py` | `fama_french_5`, `linear_mv` baselines |
| `Asset_pricing.ipynb` | Orchestration notebook — imports and calls the modules above |

## Results

Best hyperparameters found by Optuna (20 trials): `hidden_dim=32`, `lstm_layers=3`, `hidden_layer=64`, `d_lstm_hidden_dim=32`, `d_lstm_layers=2`, `epochs=500`, `d_lr=5.6e-4`, `g_lr=9.9e-4`.

### Sharpe Ratio

|  | **GAN (Ensemble, 3 runs)** | **Fama-French 5** | **Mean-Variance** |
|--|--|--|--|
| **Train** | 3.49 ± 0.05 | 0.03 ± 0.01 | 0.07 ± 0.01 |
| **Test**  | 0.51 ± 0.05 | -0.06 ± 0.52 | -0.03 ± 0.24 |

### GAN Model — Additional Metrics

|  | **Cross-sectional R²** | **Mean Abs Pricing Error** |
|--|--|--|
| **Train** | 0.008 ± 0.009 | 0.009 ± 0.003 |
| **Test**  | 0.054 ± 0.045 | 0.109 ± 0.049 |

The GAN delivers a substantially higher test Sharpe (0.51) than both the Fama-French and mean-variance baselines, which show near-zero or negative out-of-sample Sharpe ratios. The low cross-sectional R² is consistent with prior literature on reduced-form GAN-based SDF estimation with a small set of test assets — the model prices the moment conditions well but explains limited cross-sectional variation in average returns.
