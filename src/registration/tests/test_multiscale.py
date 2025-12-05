"""
Test suite for Multiscale Inverse Compositional Algorithm
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np

from ..single_scale import InverseCompositional
from ..transformations import PlanarTransform
from ..gaussian_pyramid import GaussianPyramid
from ..gradients import Gradients
from ..multiscale_registration import MultiscaleIC


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_image(size=256, device='cpu'):
    try:
        from sklearn.datasets import load_sample_image
        img = load_sample_image('china.jpg').copy()
        h, w = img.shape[:2]
        s = min(h, w)
        img = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
        return img.to(device)
    except:
        img = torch.zeros(1, 3, size, size, device=device)
        for i in range(0, size, 16):
            for j in range(0, size, 16):
                if ((i // 16) + (j // 16)) % 2 == 0:
                    img[0, :, i:i+16, j:j+16] = 0.9
                else:
                    img[0, :, i:i+16, j:j+16] = 0.3
        return img


def compute_epe(T_est, T_gt, H, W, device):
    """Compute end-point error between estimated and ground truth transforms."""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    xx_gt, yy_gt = T_gt.transform_points(xx, yy)
    xx_est, yy_est = T_est.transform_points(xx, yy)
    return torch.sqrt((xx_gt - xx_est)**2 + (yy_gt - yy_est)**2)


def create_single_scale_ic(device, C=1):
    """Create single-scale IC solver."""
    grad = Gradients(method='farid5', C=C, device=device)
    return InverseCompositional(
        transform_type='homography',
        gradient_method=grad,
        error_function='lorentzian',
        delta=5,
        epsilon=1e-3,
        max_iter=200,
        lambda_init=80.0,
        lambda_min=5.0,
        lambda_decay=0.9
    )


def create_multiscale_ic(device, C=1, min_size=32):
    """Create multiscale IC solver."""
    ic = create_single_scale_ic(device, C)
    pyramid = GaussianPyramid(
        eta=0.5,
        sigma_0=0.6,
        ksize_factor=8,
        min_size=min_size
    )
    return MultiscaleIC(
        singleScaleIC=ic,
        gaussianPyramid=pyramid,
        first_scale=0,
        grayscale=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_basic():
    """Basic functionality test."""
    print("\n" + "="*60)
    print(" BASIC MULTISCALE IC TEST ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    size = 256
    I2 = load_test_image(size, device)
    H, W = size, size
    
    # Ground truth homography
    T_gt = PlanarTransform('homography',
                           params=[0.08, 0.03, 15.0, -0.03, 0.06, 12.0, 0.0002, -0.0001],
                           device=device)
    I1 = T_gt.warp(I2)
    
    # Create and run multiscale IC
    ms_ic = create_multiscale_ic(device)
    
    start = time.perf_counter()
    result = ms_ic.run(I1, I2, return_all_scales=True)
    elapsed = time.perf_counter() - start
    
    T_est = result['transform']
    
    print(f"\nPyramid scales: {result['n_scales']}")
    print(f"Pyramid sizes: {result['pyramid_sizes']}")
    print(f"Time: {elapsed*1000:.2f} ms")
    
    # Compute EPE
    epe = compute_epe(T_est, T_gt, H, W, device)
    print(f"Mean EPE: {epe.mean():.6f} px")
    print(f"Max EPE:  {epe.max():.6f} px")
    
    return T_est, T_gt, I1, I2, epe


def test_single_vs_multiscale():
    """Compare single-scale vs multiscale performance."""
    print("\n" + "="*60)
    print(" SINGLE-SCALE vs MULTISCALE COMPARISON ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    size = 256
    I2 = load_test_image(size, device)
    H, W = size, size
    
    # Larger displacement to show multiscale benefit
    T_gt = PlanarTransform('homography',
                           params=[0.08, 0.03, 15.0, -0.03, 0.06, 12.0, 0.0002, -0.0001],
                           device=device)
    I1 = T_gt.warp(I2)
    
    # Convert to grayscale for fair comparison
    I1_gray = 0.299 * I1[:, 0:1] + 0.587 * I1[:, 1:2] + 0.114 * I1[:, 2:3]
    I2_gray = 0.299 * I2[:, 0:1] + 0.587 * I2[:, 1:2] + 0.114 * I2[:, 2:3]
    
    # Single-scale
    ss_ic = create_single_scale_ic(device, C=1)
    
    start = time.perf_counter()
    T_ss = ss_ic.run(I1_gray, I2_gray)
    ss_time = time.perf_counter() - start
    
    epe_ss = compute_epe(T_ss, T_gt, H, W, device)
    
    print(f"\nSingle-scale:")
    print(f"  Time: {ss_time*1000:.2f} ms")
    print(f"  Mean EPE: {epe_ss.mean():.6f} px")
    print(f"  Max EPE:  {epe_ss.max():.6f} px")
    
    # Multiscale
    ms_ic = create_multiscale_ic(device)
    
    start = time.perf_counter()
    T_ms = ms_ic.run(I1, I2)
    ms_time = time.perf_counter() - start
    
    epe_ms = compute_epe(T_ms, T_gt, H, W, device)
    
    print(f"\nMultiscale:")
    print(f"  Time: {ms_time*1000:.2f} ms")
    print(f"  Mean EPE: {epe_ms.mean():.6f} px")
    print(f"  Max EPE:  {epe_ms.max():.6f} px")
    
    if epe_ms.mean() > 0:
        print(f"\nImprovement: {epe_ss.mean() / epe_ms.mean():.2f}x better accuracy")
    
    return T_ss, T_ms, epe_ss, epe_ms, I1, I2


def test_displacement_robustness():
    """Test robustness to increasing displacement magnitude."""
    print("\n" + "="*60)
    print(" DISPLACEMENT MAGNITUDE TEST ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    size = 256
    I2 = load_test_image(size, device)
    H, W = size, size
    
    I2_gray = 0.299 * I2[:, 0:1] + 0.587 * I2[:, 1:2] + 0.114 * I2[:, 2:3]
    
    translations = [5, 10, 15, 20, 30, 40, 50]
    
    ss_ic = create_single_scale_ic(device, C=1)
    ms_ic = create_multiscale_ic(device)
    
    ss_epes = []
    ms_epes = []
    
    for tx in translations:
        T_gt = PlanarTransform('homography',
                               params=[0.05, 0.02, float(tx), -0.02, 0.04, float(tx)*0.8, 0.0001, -0.0001],
                               device=device)
        I1 = T_gt.warp(I2)
        I1_gray = 0.299 * I1[:, 0:1] + 0.587 * I1[:, 1:2] + 0.114 * I1[:, 2:3]
        
        # Single-scale
        T_ss = ss_ic.run(I1_gray, I2_gray)
        epe_ss = compute_epe(T_ss, T_gt, H, W, device).mean().item()
        
        # Multiscale
        T_ms = ms_ic.run(I1, I2)
        epe_ms = compute_epe(T_ms, T_gt, H, W, device).mean().item()
        
        ss_epes.append(epe_ss)
        ms_epes.append(epe_ms)
        
        ratio = epe_ss / max(epe_ms, 1e-6)
        print(f"Translation {tx:>2}px: SS={epe_ss:.4f}, MS={epe_ms:.4f}, ratio={ratio:.2f}x")
    
    return translations, ss_epes, ms_epes


def test_pyramid_visualization():
    """Visualize pyramid levels."""
    print("\n" + "="*60)
    print(" PYRAMID VISUALIZATION ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    size = 256
    I = load_test_image(size, device)
    
    pyramid = GaussianPyramid(eta=0.5, sigma_0=0.6, ksize_factor=8, min_size=32)
    pyr = pyramid(I)
    
    print(f"Number of scales: {len(pyr)}")
    for i, level in enumerate(pyr):
        print(f"  Level {i}: {level.shape[2]}x{level.shape[3]}")
    
    return pyr


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_results(T_ss, T_ms, T_gt, epe_ss, epe_ms, I1, I2, pyr):
    """Create comprehensive visualization."""
    device = I1.device
    H, W = I1.shape[2], I1.shape[3]
    
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Images
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(I1[0].permute(1, 2, 0).cpu().clamp(0, 1))
    ax1.set_title('I₁ (reference)'); ax1.axis('off')
    
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.imshow(I2[0].permute(1, 2, 0).cpu().clamp(0, 1))
    ax2.set_title('I₂ (target)'); ax2.axis('off')
    
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(T_ms.warp(I2)[0].permute(1, 2, 0).cpu().clamp(0, 1))
    ax3.set_title('I₂(Ψ(·; p̂)) multiscale'); ax3.axis('off')
    
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.imshow(T_ss.warp(I2)[0].permute(1, 2, 0).cpu().clamp(0, 1))
    ax4.set_title('I₂(Ψ(·; p̂)) single-scale'); ax4.axis('off')
    
    # Row 2: EPE maps
    vmax = max(epe_ms.max().item(), epe_ss.max().item())
    
    ax5 = fig.add_subplot(3, 4, 5)
    im5 = ax5.imshow(epe_ms.cpu(), cmap='hot', vmin=0, vmax=vmax)
    ax5.set_title(f'EPE multiscale (mean={epe_ms.mean():.4f})'); ax5.axis('off')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(epe_ss.cpu(), cmap='hot', vmin=0, vmax=vmax)
    ax6.set_title(f'EPE single-scale (mean={epe_ss.mean():.4f})'); ax6.axis('off')
    plt.colorbar(im6, ax=ax6)
    
    # Residuals
    residual_ms = (T_ms.warp(I2) - I1).abs().mean(dim=1)[0]
    residual_ss = (T_ss.warp(I2) - I1).abs().mean(dim=1)[0]
    
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.imshow(residual_ms.cpu(), cmap='gray', vmin=0, vmax=0.2)
    ax7.set_title('Residual (multiscale)'); ax7.axis('off')
    
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.imshow(residual_ss.cpu(), cmap='gray', vmin=0, vmax=0.2)
    ax8.set_title('Residual (single-scale)'); ax8.axis('off')
    
    # Row 3: Pyramid visualization
    ax9 = fig.add_subplot(3, 4, 9)
    composite = torch.zeros(1, 3, H, W, device=device)
    x_offset = 0
    for i, level in enumerate(pyr):
        h, w = level.shape[2], level.shape[3]
        if x_offset + w <= W:
            # Expand grayscale to RGB if needed
            if level.shape[1] == 1:
                level = level.expand(-1, 3, -1, -1)
            composite[0, :, :h, x_offset:x_offset+w] = level[0]
            x_offset += w + 2
    ax9.imshow(composite[0].permute(1, 2, 0).cpu().clamp(0, 1))
    ax9.set_title(f'Gaussian Pyramid ({len(pyr)} scales)'); ax9.axis('off')
    
    # Matrices comparison
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.axis('off')
    matrix_text = f"Ground Truth:\n{T_gt.matrix.cpu().numpy()}\n\n"
    matrix_text += f"Multiscale:\n{T_ms.matrix.cpu().numpy()}\n\n"
    matrix_text += f"Single-scale:\n{T_ss.matrix.cpu().numpy()}"
    ax10.text(0.1, 0.5, matrix_text, fontsize=8, family='monospace',
              verticalalignment='center', transform=ax10.transAxes)
    ax10.set_title('Transformation Matrices')
    
    plt.tight_layout()
    plt.savefig('multiscale_ic_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: multiscale_ic_results.png")


def visualize_displacement_test(translations, ss_epes, ms_epes):
    """Plot displacement robustness results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(translations, ss_epes, 'r-o', linewidth=2, markersize=8, label='Single-scale')
    ax.semilogy(translations, ms_epes, 'b-s', linewidth=2, markersize=8, label='Multiscale')
    ax.set_xlabel('Translation (pixels)', fontsize=12)
    ax.set_ylabel('Mean EPE (pixels, log scale)', fontsize=12)
    ax.set_title('Single-scale vs Multiscale: Large Displacement Robustness', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiscale_displacement_test.png', dpi=150)
    plt.show()
    print("\nSaved: multiscale_displacement_test.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run tests
    T_est, T_gt, I1, I2, epe = test_basic()
    T_ss, T_ms, epe_ss, epe_ms, I1, I2 = test_single_vs_multiscale()
    translations, ss_epes, ms_epes = test_displacement_robustness()
    pyr = test_pyramid_visualization()
    
    # Visualize
    visualize_results(T_ss, T_ms, T_gt, epe_ss, epe_ms, I1, I2, pyr)
    print('last')
    visualize_displacement_test(translations, ss_epes, ms_epes)
    
    print("\n" + "="*60)
    print(" ALL TESTS COMPLETED ")
    print("="*60)