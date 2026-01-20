"""
The objective of this file is to:
- Study the correlation of the l2 and CEMD distances
- Plot the coverage function
"""

from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


#### PARAMETERS ####

## PLOTTING ##
import numpy as np

epsilons = np.linspace(0, 50, 100)

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

featureMatching_params_1 = featureMatchingParameters(
    metric          = "L2",
    epsilon         = 1e-5,     # Will be overriden
    reciprocal_only = False,
    partial_output  = False,
    distribution='lognormal'
)

featureMatching_params_2 = featureMatchingParameters(
    metric          = "CEMD",
    epsilon         = 1,       # ! Selectivity
    reciprocal_only = False,
    partial_output  = False,
    distribution='lognormal'
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

# Convert epsilons to quantiles
quantiles = epsilons / len(histograms) / len(histograms)

print(f"Total histograms: {len(histograms)}")

##-- Compute the matches for B --##

feature_matcher_b = featureMatching(featureMatching_params_2)

matches_b, deltas_b, D_b = feature_matcher_b.match(
    query_histograms=histograms,
    key_histograms=histograms
)

print(f"Matches in B (CEMD): {len(matches_b)}")

##-- Compute the matches for A --##

feature_matcher_a = featureMatching(featureMatching_params_1)

_, _, D_a = feature_matcher_a.match(
    query_histograms=histograms,
    key_histograms=histograms
)

print(f"Distance matrix shape: {D_a.shape}")
print(f"D_a range: [{D_a.min():.6f}, {D_a.max():.6f}]")
print(f"D_b range: [{D_b.min():.6f}, {D_b.max():.6f}]")

# Compute correlation coefficient between D_a and D_b
D_a_flat = D_a.flatten()
D_b_flat = D_b.flatten()
correlation = torch.corrcoef(torch.stack([D_a_flat, D_b_flat]))[0, 1].item()
print(f"\nCorrelation between D_a (L2) and D_b (CEMD): {correlation:.4f}")

# Compute the coverage and size of A for each quantile
print("\nComputing coverage for each quantile...")
len_b = len(matches_b)
coverage_ratios = []
num_matches_a = []

for quantile in tqdm.tqdm(quantiles):
    deltas_a = feature_matcher_a.compute_delta(D_a, quantile)
    matches_a = D_a <= deltas_a.unsqueeze(1)
    matches_a_indices = torch.nonzero(matches_a, as_tuple=False)
    
    matches_a_lin = matches_a_indices[:, 0] * D_a.shape[1] + matches_a_indices[:, 1]
    matches_b_lin = matches_b[:, 0] * D_a.shape[1] + matches_b[:, 1]
    
    b_in_a = torch.isin(matches_b_lin, matches_a_lin)
    common_matches = b_in_a.sum().item()
    
    # Coverage (Recall): what fraction of B's matches are in A?
    coverage_ratio = common_matches / len_b
    coverage_ratios.append(coverage_ratio)
    
    # Number of matches in A
    num_matches_a.append(len(matches_a_indices))

# Compute absolute number of common matches (A ∩ B)
num_common = [int(coverage * len_b) for coverage in coverage_ratios]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Number of common elements vs size of A (parametric in epsilon)
ax1 = axes[0]
ax1.plot(num_matches_a, num_common, linewidth=2.5, color='#2E86AB')
# Diagonal line represents perfect precision (every element in A is also in B)
max_val = max(max(num_matches_a), len_b)
ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect precision')
ax1.axhline(y=len_b, color='green', linestyle='--', alpha=0.5, linewidth=2, label=f'|B| = {len_b:,}')
ax1.fill_between(num_matches_a, 0, num_common, alpha=0.2, color='#2E86AB')
ax1.set_xlabel('|A| (Number of elements in A)', fontsize=12)
ax1.set_ylabel('|A ∩ B| (Common elements)', fontsize=12)
ax1.set_title('Coverage vs Match Set Size (parametric in ε)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11)

# Plot 2: Fraction of B in A vs epsilon
ax2 = axes[1]
ax2.plot(epsilons, coverage_ratios, linewidth=2.5, color='#F18F01')
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='B ⊆ A (complete coverage)')
ax2.fill_between(epsilons, 0, coverage_ratios, alpha=0.2, color='#F18F01')
ax2.set_xlabel('Epsilon (ε)', fontsize=12)
ax2.set_ylabel('|B ∩ A| / |B|', fontsize=12)
ax2.set_title('Fraction of B covered by A vs ε', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(epsilons[0], epsilons[-1])
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11)

plt.suptitle(f'L2 vs CEMD Coverage Analysis | |B| = {len_b:,} | Correlation(D_L2, D_CEMD) = {correlation:.4f}', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig('coverage_plot.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Coverage plot saved to coverage_plot.png")

# Print summary statistics
print(f"\n{'='*60}")
print(f"SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Total histograms (patches): {len(histograms):,}")
print(f"Matches in B (CEMD, ε={featureMatching_params_2.epsilon}): {len_b:,}")
print(f"Correlation between D_a (L2) and D_b (CEMD): {correlation:.4f}")

print(f"\n{'─'*60}")
print(f"COVERAGE AT KEY EPSILON VALUES")
print(f"{'─'*60}")
for idx in [0, len(epsilons)//4, len(epsilons)//2, 3*len(epsilons)//4, -1]:
    eps = epsilons[idx]
    print(f"ε = {eps:.4f}: |A| = {num_matches_a[idx]:,}, |A∩B| = {num_common[idx]:,}, Coverage = {coverage_ratios[idx]:.2%}")

print(f"\n{'='*60}\n")