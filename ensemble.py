import numpy as np
from model import Generator, Discriminator
from train import training_fn
from evaluate import evaluate_GAN


def run_ensemble(params, n_rounds, train_loader, test_loader, regularize=False):
    """
    Train n_rounds independent GAN models and aggregate metrics.

    Parameters
    ----------
    params : dict with keys hidden_dim, lstm_layers, hidden_layer, epochs, d_lr, g_lr
    n_rounds : int
    train_loader : DataLoader
    test_loader : DataLoader
    regularize : bool

    Returns
    -------
    all_train_results, all_test_results : dicts mapping metric -> list of per-round values
    """
    keys = ['sharpe_ratio', 'cross_sectional_r2', 'mean_abs_pricing_error']
    all_train = {k: [] for k in keys}
    all_test = {k: [] for k in keys}

    # Infer dims from first batch
    sample = next(iter(train_loader))
    macro_dim = sample['macro_X'].shape[2]
    ff_dim = sample['ff_X'].shape[1]
    num_assets = sample['target_Y'].shape[1]

    for round_num in range(n_rounds):
        print(f"  Round {round_num + 1}/{n_rounds}")

        generator = Generator(macro_dim, ff_dim, params['hidden_dim'], params['lstm_layers'], num_assets)
        discriminator = Discriminator(params['hidden_dim'], ff_dim, num_assets, params['hidden_layer'])

        training_fn(
            generator, discriminator, train_loader,
            params['epochs'], params['d_lr'], params['g_lr'],
            regularize=regularize,
        )

        train_res = evaluate_GAN(generator, train_loader)
        test_res = evaluate_GAN(generator, test_loader)

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
