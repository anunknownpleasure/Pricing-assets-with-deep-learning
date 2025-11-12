# Pricing-assets-with-deep-learning

This project explores the application of a Generative Adversarial Network (GAN) for estimating the Stochastic Discount Factor (SDF) in asset pricing, inspired by [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3350138) paper of Chen, Pelger, and Zhu. The core objective is to learn an SDF that satisfies the no-arbitrage condition, $E[M_t R^e_{t,i}] = 0$, where $M_t$ is the SDF and $R^e_{t,i}$ are excess asset returns.

The approach utilizes a Generative Adversarial Network(GAN) consisting of a generator (SDF network) and a discriminator (conditioning network). The generator takes macroeconomic time series (processed by an LSTM) and Fama-French factors as input to output portfolio weights, which define the SDF. The fiscriminator uses the LSTM's hidden state and Fama-French factors to generate conditioning instruments used in the no-arbitrage condition. The training process involves an adversarial game where the discriminator aims to maximize the pricing errors while the generator attempts to minimize them.

For data, the project employs a reduced set compared to the original paper, using 5 macroeconomic indicators and 5 Fama-French factors as inputs. The test assets are the 25 Fama-French portfolios sorted by size and book-to-market. 

We tune the hyperparameters to identify combinations of hidden layers and learning rate based on the validation set Sharpe ratio. To mitigate variance sensitivity, we use an ensemble model which is trained multiple (15) and the results are averaged.

The evaluation compares the ensembled GAN model against traditional asset pricing baselines: the Fama-French 5-factor model and a linear mean-variance portfolio. Key performance indicators include Sharpe ratio, cross-sectional R², and mean absolute pricing error.

**Results:**

*   **Sharpe Ratio:** The ensembled GAN model demonstrates a significantly higher average Sharpe Ratio on both the training (0.79) and the combined validation/test (0.12) datasets compared to the Fama-French (Train: 0.22, Test: -0.81) and Mean-Variance (Train: 0.40, Test: -0.71) baseline models. This suggests that the GAN-derived SDF is more effective at generating risk-adjusted returns out-of-sample.

*   **Cross-sectional R²:** The GAN model's Cross-sectional R² values (Train: -0.45, Test: -0.53) indicate a limited ability to explain the cross-section of average returns in this specific implementation.

*   **Mean Absolute Pricing Error:** The GAN model exhibits a relatively low Mean Absolute Pricing Error on the training set (0.045), suggesting it can effectively price the training assets according to the no-arbitrage condition. The error is higher on the test set (0.085).
