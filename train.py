import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from model import GAN_loss


def pretrain_generator(generator, train_loader, epochs=50, lr=1e-3):
    """
    Pre-train the Generator alone to maximise the Sharpe ratio of the
    SDF portfolio before the adversarial phase begins.

    This gives the Generator a sensible starting point so the Discriminator
    cannot exploit a randomly-initialised Generator in the early epochs.

    Returns
    -------
    losses : list of per-epoch average Sharpe-maximisation loss
    """
    optimiser = optim.Adam(generator.parameters(), lr=lr)
    scheduler = ExponentialLR(optimiser, gamma=0.99)
    losses = []

    generator.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            macro_X = batch['macro_X']
            ff_X    = batch['ff_X']
            returns = batch['target_Y']

            optimiser.zero_grad()
            weights, _ = generator(macro_X, ff_X)

            portfolio_returns = (weights * returns).sum(dim=1)
            # Minimise negative Sharpe (maximise Sharpe)
            loss = -(portfolio_returns.mean() / (portfolio_returns.std() + 1e-8))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg = epoch_loss / num_batches
        losses.append(avg)

        if (epoch + 1) % 10 == 0:
            print(f'  [Pretrain] Epoch [{epoch+1}/{epochs}], Loss: {avg:.6f}')

    return losses


def training_fn(
    generator,
    discriminator,
    train_loader,
    epochs=200,
    d_lr=1e-4,
    g_lr=1e-4,
    d_to_g_ratio=2,
    regularize=True,
    lr_gamma=0.995,
):
    """
    Adversarial training loop with ExponentialLR decay on both optimisers.

    Discriminator is updated d_to_g_ratio times per generator update.
    Gradient clipping (max_norm=1.0) is applied to both networks.

    Parameters
    ----------
    lr_gamma : float
        Multiplicative LR decay factor per epoch (e.g. 0.995 decays
        the LR by ~63% over 200 epochs).

    Returns
    -------
    g_losses, d_losses : lists of per-epoch average losses
    """
    D_optim = optim.Adam(discriminator.parameters(), lr=d_lr)
    G_optim = optim.Adam(generator.parameters(), lr=g_lr)

    D_scheduler = ExponentialLR(D_optim, gamma=lr_gamma)
    G_scheduler = ExponentialLR(G_optim, gamma=lr_gamma)

    g_losses = []
    d_losses = []

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            macro_X = batch['macro_X']
            ff_X    = batch['ff_X']
            returns = batch['target_Y']

            # --- Train Discriminator ---
            for _ in range(d_to_g_ratio):
                D_optim.zero_grad()
                with torch.no_grad():
                    weights, _ = generator(macro_X, ff_X)
                g_inst = discriminator(macro_X, ff_X)
                d_loss = GAN_loss.discriminator_loss(weights, returns, g_inst, reg=regularize)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                D_optim.step()

            # --- Train Generator ---
            G_optim.zero_grad()
            weights, _ = generator(macro_X, ff_X)
            with torch.no_grad():
                g_inst = discriminator(macro_X, ff_X)
            g_loss = GAN_loss.generator_loss(weights, returns, g_inst, reg=regularize)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            G_optim.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        D_scheduler.step()
        G_scheduler.step()

        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], G Loss: {avg_g:.6f}, D Loss: {avg_d:.6f}')

    return g_losses, d_losses
