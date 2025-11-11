import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from model import GAN_loss 

def training_fn(generator, discriminator, G_optim, D_optim, train_loader, criterion, epochs):
    """
    Corrected training function.
    Accepts optimizers and criterion as arguments.
    """
    
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        # Set models to training mode (e.g., for Dropout, BatchNorm)
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0

        for batch in train_loader:
            macro_X = batch['macro_X']
            ff_X = batch['ff_X']
            returns = batch['target_Y']

            # --- Training the discriminator ---
            # We train it more (k=2) to keep it ahead of the generator
            for _ in range(2):
                D_optim.zero_grad()

                # Get generator outputs
                weights, h_t = generator(macro_X, ff_X)
                
                # Get discriminator's portfolio
                # We detach h_t so gradients don't flow back to the generator
                g_inst = discriminator(h_t.detach(), ff_X.detach())

                # Compute discriminator loss
                # We detach weights so D isn't trying to update G
                g_loss, d_loss = criterion(weights.detach(), returns, g_inst)

                d_loss.backward()
                D_optim.step()

            # --- Training the generator ---
            G_optim.zero_grad()

            # Get generator outputs
            # ***CRITICAL: Use the real, non-detached h_t***
            weights, h_t = generator(macro_X, ff_X)

            # Get discriminator's portfolio
            # ***CRITICAL: Use the non-detached h_t***
            g_inst = discriminator(h_t, ff_X)

            # Compute generator loss
            # g_inst is detached inside the loss function, 
            # or we can detach it here
            g_loss, d_loss = criterion(weights, returns, g_inst.detach())

            g_loss.backward()
            G_optim.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

        # Average losses in an epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], G Loss: {avg_g_loss:.6f}, D Loss: {avg_d_loss:.6f}')

    return g_losses, d_losses



def eval_fn(generator, discriminator, test_loader, criterion):
  
    generator.eval()
    discriminator.eval()

    all_sdf_m = []
    all_returns_e = []
    all_sdf_portfolio_returns = []
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_test_asset_return_std = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            macro = batch['macro_X']
            char = batch['ff_X']
            returns = batch['target_Y'] # These are R_e (excess returns)

            # --- Get Model Outputs ---
            omega, h_t = generator(macro, char)
            g_inst = discriminator(h_t, char)
            
            # --- 1. Calculate Adversarial Loss (Pricing Error) ---
            g_loss, d_loss = criterion(omega, returns, g_inst)
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # Need the raw test asset return to calculate Sharpe
            test_asset_return = torch.sum(g_inst * returns, dim=1)
            total_test_asset_return_std += torch.std(test_asset_return).item()
            
            # --- 2. Calculate E[M * R_e] ---
            sdf_portfolio_return = (omega * returns).sum(dim=1)
            sdf_m = 1.0 - sdf_portfolio_return
            
            all_sdf_m.append(sdf_m.numpy())
            all_returns_e.append(returns.numpy())
            
            # --- 3. Calculate SDF Portfolio Sharpe Ratio ---
            all_sdf_portfolio_returns.append(sdf_portfolio_return.numpy())
            
            num_batches += 1

    # --- Aggregate Metrics ---
    results = {}
    
    # 1. Adversarial Sharpe Ratio (How "hard" was the test set?)
    # This is the "Generator Loss" from your notebook's eval_fn
    avg_g_loss = total_g_loss / num_batches
    avg_std = total_test_asset_return_std / num_batches
    results['adversarial_sharpe_ratio'] = (avg_g_loss / (avg_std + 1e-8)) * np.sqrt(12)

    # 2. SDF Portfolio Sharpe Ratio
    sdf_portfolio_returns = np.concatenate(all_sdf_portfolio_returns)
    results['sharpe_ratio'] = (sdf_portfolio_returns.mean() / sdf_portfolio_returns.std()) * np.sqrt(12)

    # 3. Mean Absolute Pricing Error (E[M * R_e])
    sdf_m = np.concatenate(all_sdf_m)
    returns_e = np.concatenate(all_returns_e)
    pricing_errors = (sdf_m.reshape(-1, 1) * returns_e).mean(axis=0) # E[M*R_e] for each asset
    results['mean_abs_pricing_error'] = np.abs(pricing_errors).mean()

    # 4. Cross-sectional R² (Corrected)
    # This regresses actual mean returns (Y) on predicted betas (X)
    try:
        # Y = E[R_e]
        Y = returns_e.mean(axis=0)
        
        # X = Beta = Cov(R_e, M) / Var(M)
        # We approximate M with the SDF portfolio return (R_sdf)
        # So Beta = Cov(R_e, R_sdf) / Var(R_sdf)
        # Note: Using R_sdf (portfolio_returns) instead of M (sdf_m) is standard
        X = np.array([np.cov(returns_e[:, i], sdf_portfolio_returns)[0, 1] for i in range(returns_e.shape[1])])
        X = X / np.var(sdf_portfolio_returns)
        
        # Add intercept
        X = np.c_[np.ones(X.shape[0]), X]
        
        # OLS: (X'X)^-1 * X'Y
        betas = np.linalg.inv(X.T @ X) @ X.T @ Y
        y_pred = X @ betas
        
        ss_res = np.sum((Y - y_pred)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        results['cross_sectional_r2'] = 1 - (ss_res / ss_tot)
    except Exception as e:
        # print(f"Could not calculate R2_CS: {e}")
        results['cross_sectional_r2'] = -np.inf # Failed

    return results