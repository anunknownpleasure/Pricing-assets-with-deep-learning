import torch
import torch.optim as optim
from model import GAN_loss


def training_fn(
    generator,
    discriminator,
    train_loader,
    epochs,
    d_lr,
    g_lr,
    d_to_g_ratio=2,
    regularize=True,
):
    """
    Adversarial training loop.

    Discriminator is updated d_to_g_ratio times per generator update.
    Gradient clipping (max_norm=1.0) is applied to both networks.

    Returns
    -------
    g_losses, d_losses : lists of per-epoch average losses
    """
    D_optim = optim.Adam(discriminator.parameters(), lr=d_lr)
    G_optim = optim.Adam(generator.parameters(), lr=g_lr)

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
            ff_X = batch['ff_X']
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

        avg_g = epoch_g_loss / num_batches
        avg_d = epoch_d_loss / num_batches
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], G Loss: {avg_g:.6f}, D Loss: {avg_d:.6f}')

    return g_losses, d_losses
