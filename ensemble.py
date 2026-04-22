import numpy as np
from model import Generator, Discriminator
from train import pretrain_generator, training_fn
from evaluate import evaluate_GAN


def run_ensemble(params, n_rounds, train_loader, test_loader, regularize=False):
    """
    Train n_rounds independent GAN models and aggregate metrics.

    Each round:
      1. Pre-train Generator alone (Sharpe maximisation) for pretrain_epochs
      2. Adversarial training with LR decay for params['epochs']

    Parameters
    ----------
    params : dict with keys hidden_dim, lstm_layers, hidden_layer, epochs,
             d_lr, g_lr, and optionally pretrain_epochs (default 50),
             d_lstm_hidden_dim (default 16), d_lstm_layers (default 1)
    """
    keys = ['sharpe_ratio', 'cross_sectional_r2', 'mean_abs_pricing_error']
    all_train = {k: [] for k in keys}
    all_test  = {k: [] for k in keys}

    sample = next(iter(train_loader))
    macro_dim  = sample['macro_X'].shape[2]
    ff_dim     = sample['ff_X'].shape[1]
    num_assets = sample['target_Y'].shape[1]

    pretrain_epochs = params.get('pretrain_epochs', 50)

    for round_num in range(n_rounds):
        print(f"  Round {round_num + 1}/{n_rounds}")

        generator = Generator(macro_dim, ff_dim, params['hidden_dim'],
                              params['lstm_layers'], num_assets)
        discriminator = Discriminator(macro_dim, ff_dim, num_assets,
                                      params['hidden_layer'],
                                      d_lstm_hidden_dim=params.get('d_lstm_hidden_dim', 16),
                                      d_lstm_layers=params.get('d_lstm_layers', 1))

        pretrain_generator(generator, train_loader, epochs=pretrain_epochs, lr=params['g_lr'])

        training_fn(
            generator, discriminator, train_loader,
            params['epochs'], params['d_lr'], params['g_lr'],
            regularize=regularize,
        )

        train_res = evaluate_GAN(generator, train_loader)
        test_res  = evaluate_GAN(generator, test_loader)

        for k in keys:
            all_train[k].append(train_res[k])
            all_test[k].append(test_res[k])

    return all_train, all_test


def print_ensemble_summary(all_train, all_test):
    print("\nEnsembled GAN Results Summary:")
    print("-" * 40)
    for metric in ['sharpe_ratio', 'cross_sectional_r2', 'mean_abs_pricing_error']:
        avg_train = np.mean(all_train[metric])
        std_train = np.std(all_train[metric])
        avg_test = np.mean(all_test[metric])
        std_test = np.std(all_test[metric])
        label = metric.replace('_', ' ').title()
        print(f"Metric: {label}")
        print(f"  Average Train: {avg_train:.4f} (Std Dev: {std_train:.4f})")
        print(f"  Average Test:  {avg_test:.4f} (Std Dev: {std_test:.4f})")
        print("-" * 20)
