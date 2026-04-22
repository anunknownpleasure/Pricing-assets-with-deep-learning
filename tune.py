import torch
import numpy as np
import random
import optuna

from model import Generator, Discriminator
from train import pretrain_generator, training_fn
from evaluate import evaluate_GAN


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_optuna_study(train_loader, val_loader, n_trials=50):
    """
    Tune Generator/Discriminator hyperparameters with Optuna,
    maximising validation Sharpe ratio.

    Each trial pre-trains the Generator for 50 epochs then runs
    the adversarial phase with LR decay.

    Uses trial.number as the random seed for reproducible diversity.

    Returns
    -------
    study : optuna.Study
    best_params : dict
    """
    sample = next(iter(train_loader))
    macro_dim  = sample['macro_X'].shape[2]
    ff_dim     = sample['ff_X'].shape[1]
    num_assets = sample['target_Y'].shape[1]

    def objective(trial):
        _set_seed(trial.number)

        hidden_dim        = trial.suggest_categorical('hidden_dim',        [8, 16, 32])
        lstm_layers       = trial.suggest_categorical('lstm_layers',       [2, 3, 4])
        hidden_layer      = trial.suggest_categorical('hidden_layer',      [16, 32, 64])
        d_lstm_hidden_dim = trial.suggest_categorical('d_lstm_hidden_dim', [8, 16, 32])
        d_lstm_layers     = trial.suggest_categorical('d_lstm_layers',     [1, 2])
        epochs            = trial.suggest_categorical('epochs',            [100, 150, 200])
        d_lr = trial.suggest_float('d_lr', 1e-5, 1e-3, log=True)
        g_lr = trial.suggest_float('g_lr', 1e-5, 1e-3, log=True)

        generator = Generator(macro_dim, ff_dim, hidden_dim, lstm_layers, num_assets)
        discriminator = Discriminator(macro_dim, ff_dim, num_assets, hidden_layer,
                                      d_lstm_hidden_dim=d_lstm_hidden_dim,
                                      d_lstm_layers=d_lstm_layers)

        pretrain_generator(generator, train_loader, epochs=50, lr=g_lr)
        training_fn(generator, discriminator, train_loader, epochs, d_lr, g_lr)

        val_results = evaluate_GAN(generator, val_loader)
        return val_results['sharpe_ratio']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study, study.best_params
