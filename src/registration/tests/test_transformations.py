from ..transformations import *

# python -m src.registration.tests.test_transformations
# ═══════════════════════════════════════════════════════════════════════════
# TESTS & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def load_sample_image():
    """Load sample image from sklearn or create checkerboard."""
    try:
        from sklearn.datasets import load_sample_image
        img = load_sample_image('china.jpg').copy()  # .copy() to make writable
        h, w = img.shape[:2]
        s = min(h, w)
        img = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        return img
    except:
        # Checkerboard fallback
        img = torch.zeros(1, 3, 256, 256)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if ((i // 32) + (j // 32)) % 2 == 0:
                    img[0, :, i:i+32, j:j+32] = torch.tensor([0.9, 0.3, 0.3])[:, None, None]
                else:
                    img[0, :, i:i+32, j:j+32] = torch.tensor([0.3, 0.5, 0.9])[:, None, None]
        return img


def to_numpy(img):
    return img[0].permute(1, 2, 0).clamp(0, 1).numpy()


def test_inverse_points():
    """Test that T.inv correctly inverts point transformations."""
    print("=" * 60)
    print("TEST: Point transformation inverse")
    print("=" * 60)
    
    # Test points in center region
    xx, yy = torch.meshgrid(
        torch.linspace(50, 200, 10),
        torch.linspace(50, 200, 10),
        indexing='ij'
    )
    
    test_params = {
        'translation': [10.0, -5.0],
        'euclidean':   [10.0, 5.0, 0.1],
        'similarity':  [5.0, 5.0, 0.05, 0.02],
        'affinity':    [5.0, 5.0, 0.05, 0.02, -0.02, 0.03],
        'homography':  [0.05, 0.02, 10.0, -0.02, 0.03, 5.0, 0.0001, -0.0001],
    }
    
    for ttype, params in test_params.items():
        T = PlanarTransform(ttype, params=params)
        T_inv = T.inv
        
        # Forward then inverse on points
        xx_t, yy_t = T.transform_points(xx, yy)
        xx_r, yy_r = T_inv.transform_points(xx_t, yy_t)
        
        point_err = max((xx - xx_r).abs().max().item(), (yy - yy_r).abs().max().item())
        
        # Matrix composition check
        composed = (T @ T_inv).matrix
        matrix_err = (composed - torch.eye(3)).abs().max().item()
        
        status = "✓" if point_err < 1e-3 and matrix_err < 1e-5 else "✗"
        print(f"{status} {ttype:12s}: point_err={point_err:.2e}, matrix_err={matrix_err:.2e}")


def test_inverse_warp():
    """Test inverse warping with visualization."""
    print("\n" + "=" * 60)
    print("TEST: Image warp inverse")
    print("=" * 60)
    
    img = load_sample_image()
    H, W = img.shape[2], img.shape[3]
    
    # Small transformation to avoid boundary issues
    T = PlanarTransform('homography', 
                        params=[0.02, 0.01, 5.0, -0.01, 0.02, 3.0, 0.00005, -0.00003])
    
    # Compute valid region: where both T and T.inv keep points inside
    mask_forward = T.visibility_mask(H, W, delta=5)
    mask_inverse = T.inv.visibility_mask(H, W, delta=5)
    mask = mask_forward & mask_inverse
    
    # Warp forward then back
    warped = T.warp(img)
    recovered = T.inv.warp(warped)
    
    # Error only in valid region
    diff = (img - recovered).abs()
    diff_masked = diff * mask.float()
    
    valid_pixels = mask.sum().item()
    if valid_pixels > 0:
        max_err = diff_masked.max().item()
        mean_err = diff_masked.sum().item() / (valid_pixels * 3)  # 3 channels
    else:
        max_err = mean_err = float('nan')
    
    print(f"Valid region: {valid_pixels}/{H*W} pixels ({100*valid_pixels/(H*W):.1f}%)")
    print(f"Max error in valid region:  {max_err:.6f}")
    print(f"Mean error in valid region: {mean_err:.6f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(to_numpy(img))
    axes[0, 0].set_title('Original I')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_numpy(warped))
    axes[0, 1].set_title('Warped T(I)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(to_numpy(recovered))
    axes[0, 2].set_title('Recovered T⁻¹(T(I))')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(mask.numpy(), cmap='gray')
    axes[1, 0].set_title(f'Valid mask\n{100*valid_pixels/(H*W):.1f}% valid')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(to_numpy(diff * 10))
    axes[1, 1].set_title(f'|I - T⁻¹(T(I))| × 10')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(to_numpy(diff_masked * 10))
    axes[1, 2].set_title(f'Masked error × 10\nmax={max_err:.4f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('code_tests/test_inverse_warp.png', dpi=150)
    plt.show()
    print("Saved: test_inverse_warp.png")


def demo_all_transforms():
    """Show all transformation types."""
    print("\n" + "=" * 60)
    print("DEMO: All transformation types")
    print("=" * 60)
    
    img = load_sample_image()
    
    transforms = {
        'translation': [20.0, -15.0],
        'euclidean':   [15.0, 10.0, 0.15],
        'similarity':  [0.0, 0.0, 0.15, 0.1],
        'affinity':    [0.0, 0.0, 0.1, 0.12, -0.08, 0.05],
        'homography':  [0.08, 0.04, 15.0, -0.04, 0.06, 10.0, 0.0002, -0.0001],
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    axes[0].imshow(to_numpy(img))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, (ttype, params) in enumerate(transforms.items(), 1):
        T = PlanarTransform(ttype, params=params)
        warped = T.warp(img)
        axes[idx].imshow(to_numpy(warped))
        axes[idx].set_title(f'{ttype} (n={T.n_params})')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('code_tests/demo_all_transforms.png', dpi=150)
    plt.show()
    print("Saved: demo_all_transforms.png")


def demo_composition():
    """Verify that T1(T2(I)) == (T1 @ T2)(I)."""
    print("\n" + "=" * 60)
    print("DEMO: Composition T1 ∘ T2")
    print("=" * 60)
    
    img = load_sample_image()
    
    T1 = PlanarTransform('euclidean', params=[20.0, 0.0, 0.15])
    T2 = PlanarTransform('euclidean', params=[0.0, 15.0, -0.1])
    
    # Sequential application
    warped_seq = T1.warp(T2.warp(img))
    
    # Composed transformation
    T_composed = T1 @ T2
    warped_composed = T_composed.warp(img)
    
    diff = (warped_seq - warped_composed).abs()
    print(f"Max difference: {diff.max().item():.2e}")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(to_numpy(img))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(to_numpy(warped_seq))
    axes[1].set_title('T₁(T₂(I))')
    axes[1].axis('off')
    
    axes[2].imshow(to_numpy(warped_composed))
    axes[2].set_title('(T₁∘T₂)(I)')
    axes[2].axis('off')
    
    axes[3].imshow(to_numpy(diff * 100))
    axes[3].set_title(f'Diff ×100 (max={diff.max():.2e})')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('code_tests/demo_composition.png', dpi=150)
    plt.show()
    print("Saved: demo_composition.png")


def demo_jacobian():
    """Visualize Jacobian structure."""
    print("\n" + "=" * 60)
    print("DEMO: Jacobian visualization")
    print("=" * 60)
    
    H, W = 64, 64
    yy, xx = torch.meshgrid(
        torch.linspace(0, H-1, H),
        torch.linspace(0, W-1, W),
        indexing='ij'
    )
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, ttype in enumerate(PlanarTransform.TYPES.keys()):
        T = PlanarTransform(ttype)
        J = T.jacobian(xx, yy)
        J_mag = J.abs().sum(dim=(-1, -2))
        
        im = axes[idx].imshow(J_mag.numpy(), cmap='viridis')
        axes[idx].set_title(f'{ttype}\nshape: {tuple(J.shape)}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.suptitle('Jacobian Magnitude ||J(x,y)||₁', fontsize=14)
    plt.tight_layout()
    plt.savefig('code_tests/demo_jacobian.png', dpi=150)
    plt.show()
    print("Saved: demo_jacobian.png")


if __name__ == "__main__":
    test_inverse_points()
    test_inverse_warp()
    demo_all_transforms()
    demo_composition()
    demo_jacobian()
    
    print("\n" + "=" * 60)
    print("All tests and demos complete!")
    print("=" * 60)