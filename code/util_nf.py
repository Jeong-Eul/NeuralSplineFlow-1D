import math
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import os
import numpy as np

        
def plot_training_1d_kde(args, ssc, device, base_dist, y_original, model, epoch, treatment):
    with torch.no_grad():
        
        # Normal distribution
        z_samples = base_dist.sample((50000,)).to(device)
        
        # Flow forward
        y_vis, _ = model._elementwise_forward(z_samples.unsqueeze(-1))
        y_np = y_vis.detach().cpu().squeeze().numpy()  
        
        inversed = ssc.inverse_transform(y_np.reshape(-1, 1))

        plt.figure(figsize=(8, 4))
        plt.hist(y_original, bins=30, density=True, alpha=0.3, color='gray', label='Histogram') # 관측 히스토그램
        sns.kdeplot(inversed, label=f'P(Y[{treatment}] = y)', color='green')
        plt.title(f"Epoch {epoch}")
        plt.xlabel("y")
        plt.ylabel("Density")
        plt.legend()

        
        treatment = str(int(args.treatment))
        cluster = str(int(args.cluster_num))
        
        png_dir = f"train_plots_{args.dataset}_cluster_{cluster}_treatment_{treatment}"
        
        os.makedirs(png_dir, exist_ok=True)
        plt.savefig(png_dir + f"/epoch_{epoch:04d}.png")
        plt.close()


def std_norm_cdf(z):
    return 0.5*(1 + torch.erf(z / math.sqrt(2)))


def sum_probability(model, base_dist, device):
    
    # Sample code to estimate ∫ p_Y(y) dy over [y_min, y_max] using CPU histogram

    # 1. Sample many z from truncated normal base
    N = 20000
    z = base_dist.sample((N,)).to(device)  # shape (N, 1)

    # 2. Push through the trained flow to get y samples
    y_gen, _ = model._elementwise_forward(z.unsqueeze(-1)) # shape (N, 1)
    y_gen = y_gen.squeeze(1)     # shape (N,)

    # 3. Move to CPU for histogram
    y_cpu = y_gen.detach().cpu()

    # 4. Estimate density via histogram over [y_min, y_max]
    bins = 1000
    counts, bin_edges = torch.histogram(y_cpu, bins=bins)

    # 5. Compute bin widths (constant here)
    bin_widths = bin_edges[1:] - bin_edges[:-1]  # shape (bins,)

    # 6. Normalize histogram to get density p_Y
    density = counts.float() / (N * bin_widths)

    # 7. Approximate integral ∫ p_Y(y) dy
    integral = (density * bin_widths).sum()

    print(f"Estimated ∫ p_Y(y) dy ≈ {integral:.4f}")
    

def compute_summary_stats(arr):
    return np.mean(arr), np.var(arr)