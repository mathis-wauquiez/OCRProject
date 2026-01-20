"""
The objective of this file is to:
- Study and choose the law of the dissimilarity D(q, K), with K a RV
- 
"""

from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


#### PARAMETERS ####

## FILES ##

image_folder = Path('data/datasets/book1')
comps_folder = Path('data/extracted/book1-complete/components/')

FILELIMIT = 10

assert image_folder.exists()
assert comps_folder.exists()

## HOG ##

grayscale = True # compute the HOG on grayscale images
grdt_kernel = 'gaussian'
grdt_sigma = 2.5
cell_size = 16
patch_size = 112
num_bins = 9

## FEATURE MATCHING ##
from src.character_linking.params import HOGParameters, featureMatchingParameters, fullHOGOutput, featureMatchingOutputs

featureMatching_params_2 = featureMatchingParameters(
    metric          = "CEMD",  # ! Only parameter that matters
    epsilon         = 1,       # ? Useless here
    reciprocal_only = False,
    partial_output  = False,
    distribution='gamma'
)

## OTHER ##
device = "cuda"

#### IMPORTS ####

import torch
from einops import rearrange

import sys
import os

import cv2
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm

from src.character_linking.feature_matching import featureMatching
from src.patch_processing.hog import HOG
from src.character_linking.params import HOGParameters, featureMatchingParameters, fullHOGOutput, featureMatchingOutputs
from src.ocr.patch_extraction import extract_patches
from src.utils import connectedComponent, torch_to_pil


files = next(os.walk(image_folder))[2][:FILELIMIT]


#### MAIN CODE ####


## --- HOG --- ##


hog_params = HOGParameters(
    device          = "cuda",
    C               = 1 if grayscale else 3,
    partial_output  = False,
    method          = grdt_kernel,
    grdt_sigma      = grdt_sigma,
    ksize_factor    = 6,
    cell_height     = cell_size,
    cell_width      = cell_size,
    psize           = patch_size,
    num_bins        = num_bins,
    sigma           = None,
    threshold       = 0.2
)

hog = HOG(hog_params)

hist_list = []
patches_list = []

for file in tqdm.tqdm(files):
    img_np = np.array(Image.open(image_folder / file))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)[..., None]
    img_torch = torch.tensor(img_np, device="cuda").permute(2,0,1).to(dtype=torch.float32) / 255
    img_torch.requires_grad = False

    img_comp = connectedComponent.load(comps_folder / (str(file) + '.npz'))
    hog_output = hog(img_torch, img_comp)

    # We have one HOG descriptor per patch per color channel -> transform it to C * Ncells cells

    histograms = rearrange(hog_output.histograms, 'Npatch C Ncells Nbins -> Npatch (C Ncells) Nbins')
    hist_list.append(histograms)
    patches_list.append(hog_output.patches_image)

histograms = torch.cat(hist_list, dim=0).to(device)
patches = torch.cat(patches_list, dim=0).to(device)
del hist_list, patches_list

print(f"Total histograms: {len(histograms)}")

##-- Compute the matches for B --##

feature_matcher_b = featureMatching(featureMatching_params_2)

matches_b, deltas_b, D_b = feature_matcher_b.match(
    query_histograms=histograms,
    key_histograms=histograms
)

D_b = D_b[:, :500]

print(f"Matches in B: {len(matches_b)}")



#########
#
# D_b is (N_distrs, N_obs)
#
#
#########


"""
Distribution family testing for D_b dissimilarity matrix.
Tests which distribution family (gamma, normal, lognormal) best fits the data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def test_distribution_families(data, distributions=None):
    """
    Test multiple distribution families on (N_distrs, N_obs) data.
    
    Parameters:
    -----------
    data : np.ndarray, shape (N_distrs, N_observations)
    distributions : dict, optional
        {name: scipy.stats distribution class}
    
    Returns:
    --------
    results_df : DataFrame with comparison metrics
    """
    
    if distributions is None:
        distributions = {
            'gamma': stats.gamma,
            'lognormal': stats.lognorm,
            'weibull': stats.weibull_min,
            'alpha': stats.alpha,
            'betaprime': stats.betaprime,
            'burr': stats.burr,
            'erlang': stats.erlang,
            'exponnorm': stats.exponnorm,
            'fisk': stats.fisk,
            'exponweib': stats.exponweib,
            # 'frechet_r': stats.frechet_r,
            'genextreme': stats.genextreme,
            'invgamma': stats.invgamma,
            'invgauss': stats.invgauss,
            'invweibull': stats.invweibull,
            'mielke': stats.mielke,
            'ncf': stats.ncf,
            'normal': stats.norm,
            'rayleigh': stats.rayleigh,  # If computing Euclidean-like distances
        }
    
    N_distrs = data.shape[0]
    print(f"Testing {len(distributions)} distributions on {N_distrs} samples\n")
    
    # Check for data issues
    if np.any(data <= 0) and ('lognormal' in distributions or 'gamma' in distributions):
        print("⚠️  Warning: Data contains non-positive values")
        print("   Filtering to positive values for lognormal/gamma\n")
    
    results = []
    
    for dist_name, dist_class in distributions.items():
        print(f"Fitting {dist_name}...", end=' ')
        
        ks_stats = []
        p_values = []
        aics = []
        n_failed = 0
        
        for i in range(N_distrs):
            sample = data[i]
            
            # Filter non-positive for certain distributions
            if dist_name in ['lognormal', 'gamma']:
                sample = sample[sample > 0]
                if len(sample) < 10:  # Need minimum observations
                    n_failed += 1
                    continue
            
            try:
                # Fit parameters
                params = dist_class.fit(sample)
                
                # KS test
                ks_stat, p_val = kstest(sample, dist_class(*params).cdf)
                
                # Log-likelihood and AIC
                ll = np.sum(dist_class.logpdf(sample, *params))
                if np.isfinite(ll):
                    aic = 2 * len(params) - 2 * ll
                    
                    ks_stats.append(ks_stat)
                    p_values.append(p_val)
                    aics.append(aic)
                else:
                    n_failed += 1
                    
            except:
                n_failed += 1
        
        # Aggregate results
        n_success = len(aics)
        results.append({
            'distribution': dist_name,
            'n_fitted': n_success,
            'n_failed': n_failed,
            'mean_aic': np.mean(aics),
            'median_aic': np.median(aics),
            'mean_ks_stat': np.mean(ks_stats),
            'ks_pass_rate': np.mean(np.array(p_values) > 0.05) * 100,
        })
        
        print(f"{n_success}/{N_distrs} fitted successfully")
    
    results_df = pd.DataFrame(results).sort_values('mean_aic')
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(results_df.to_string(index=False, float_format='%.2f'))
    print("="*70)
    print(f"\n🏆 Best fit: {results_df.iloc[0]['distribution'].upper()}")
    print("="*70 + "\n")
    
    return results_df


def plot_fit_comparison(data, distributions, output_dir='/mnt/user-data/outputs/'):
    """Quick visualization of fits."""
    
    n_dists = len(distributions)
    n_cols = min(3, n_dists)
    n_rows = (n_dists + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes for easy indexing
    if n_dists == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_dists > n_cols else axes
    
    # Sample a few distributions to visualize
    sample_indices = np.random.choice(data.shape[0], size=min(5, data.shape[0]), replace=False)
    
    for idx, (dist_name, dist_class) in enumerate(distributions.items()):
        ax = axes[idx]
        
        for i in sample_indices:
            sample = data[i]
            if dist_name in ['lognormal', 'gamma', 'exponential', 'weibull', 'rayleigh', 'chi2', 'invgamma']:
                sample = sample[sample > 0]
            
            # Fit and plot
            try:
                params = dist_class.fit(sample)
                x = np.linspace(sample.min(), sample.max(), 100)
                pdf = dist_class.pdf(x, *params)
                
                ax.hist(sample, bins=30, density=True, alpha=0.3, color='gray')
                ax.plot(x, pdf, linewidth=2, alpha=0.7)
                
            except:
                pass
        
        ax.set_title(f'{dist_name.capitalize()}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    # Hide unused subplots
    for idx in range(len(distributions), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'distribution_fits.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return fig

# Add to your existing code after computing D_b:

if __name__ == "__main__":
    # Your existing code...
    # D_b is now computed as (N_distrs, N_obs)
    
    print("\n" + "="*70)
    print("TESTING DISTRIBUTION FAMILIES")
    print("="*70 + "\n")
    
    # Test distributions
    results = test_distribution_families(D_b.cpu().numpy())
    
    # Save results
    output_dir = Path('outputs/')
    output_dir.mkdir(exist_ok=True, parents=True)
    results.to_csv(output_dir / 'distribution_comparison.csv', index=False)
    
    # Visualize (optional)
    distributions = {
        'gamma': stats.gamma,
        'lognormal': stats.lognorm,
        'weibull': stats.weibull_min,
        'alpha': stats.alpha,
        'betaprime': stats.betaprime,
        'burr': stats.burr,
        'erlang': stats.erlang,
        'exponnorm': stats.exponnorm,
        'fisk': stats.fisk,
        'exponweib': stats.exponweib,
        # 'frechet_r': stats.frechet_r,
        'genextreme': stats.genextreme,
        'invgamma': stats.invgamma,
        'invgauss': stats.invgauss,
        'invweibull': stats.invweibull,
        'mielke': stats.mielke,
        'ncf': stats.ncf,
        'normal': stats.norm,
        'rayleigh': stats.rayleigh,  # If computing Euclidean-like distances
    }

    plot_fit_comparison(D_b.cpu().numpy(), distributions, output_dir)
    
    print(f"\nResults saved to {output_dir}")