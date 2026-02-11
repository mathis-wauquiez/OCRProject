"""
Batch consistency tests.

Verifies that running B pairs at once gives the same result as
running each pair individually.

    python -m registration.tests.test_batch_consistency
"""

import torch
import torch.nn.functional as F

from ..transformations import PlanarTransform
from ..gradients import Gradients
from ..gaussian_pyramid import GaussianPyramid
from ..single_scale import InverseCompositional
from ..multiscale_registration import MultiscaleIC


def load_test_image(size=64):
    """Simple checkerboard."""
    img = torch.zeros(1, 1, size, size)
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            if ((i // 8) + (j // 8)) % 2 == 0:
                img[0, 0, i:i+8, j:j+8] = 0.9
            else:
                img[0, 0, i:i+8, j:j+8] = 0.3
    return img


def test_batch_transform_ops():
    """Test PlanarTransform batched operations."""
    print("=== Batched PlanarTransform operations ===")

    B = 4
    T = PlanarTransform('homography', batch_size=B)
    assert T.matrix.shape == (B, 3, 3)
    assert torch.allclose(T.matrix[0], torch.eye(3))

    params = torch.randn(B, 8) * 0.01
    T = PlanarTransform('homography', params=params)
    assert T.batch_size == B

    # T @ T.inv ≈ I
    T2 = T @ T.inv
    err = (T2.matrix - torch.eye(3).unsqueeze(0).expand(B, -1, -1)).abs().max().item()
    assert err < 1e-5, f"T @ T.inv should ≈ I, got {err:.2e}"

    # broadcast B=1 @ B=N
    T1 = PlanarTransform('homography', batch_size=1)
    assert (T1 @ T).batch_size == B
    assert (T @ T1).batch_size == B

    # batched warp
    img = torch.randn(B, 1, 32, 32)
    warped = T.warp(img)
    assert warped.shape == (B, 1, 32, 32)

    # batched visibility mask
    mask = T.visibility_mask(32, 32, delta=2)
    assert mask.shape == (B, 32, 32)

    print("  PASSED\n")


def test_single_scale_batch_consistency():
    """B=4 batched must match 4× individual B=1."""
    print("=== Single-scale batch consistency (translation, L2) ===")

    torch.manual_seed(42)
    B, C, H, W = 4, 1, 64, 64
    dtype = torch.float64

    I1 = torch.randn(B, C, H, W, dtype=dtype)
    I2 = torch.roll(I1, shifts=(2, 3), dims=(2, 3)) + 0.01 * torch.randn(B, C, H, W, dtype=dtype)

    ic = InverseCompositional(
        transform_type='translation', gradient_method='farid5',
        error_function='l2', max_iter=20, epsilon=1e-6, dtype=dtype,
    )

    T_batch = ic.run(I1, I2)
    assert T_batch.batch_size == B

    matrices_individual = []
    for i in range(B):
        T_i = ic.run(I1[i], I2[i])
        matrices_individual.append(T_i.matrix)
    M_individual = torch.cat(matrices_individual, dim=0)

    diff = (T_batch.matrix - M_individual).abs().max().item()
    print(f"  Max diff (batch vs individual): {diff:.2e}")
    assert diff < 1e-5, f"FAIL: {diff:.2e}"
    print("  PASSED\n")


def test_single_scale_batch_consistency_robust():
    """Same with Lorentzian + homography."""
    print("=== Single-scale batch consistency (homography, Lorentzian) ===")

    torch.manual_seed(123)
    B, C, H, W = 3, 1, 48, 48
    dtype = torch.float64

    I1 = torch.randn(B, C, H, W, dtype=dtype)
    I2 = I1.clone() + 0.02 * torch.randn_like(I1)

    ic = InverseCompositional(
        transform_type='homography', gradient_method='farid5',
        error_function='lorentzian', max_iter=15, epsilon=1e-7,
        lambda_init=80.0, dtype=dtype,
    )

    T_batch = ic.run(I1, I2)
    matrices_individual = []
    for i in range(B):
        T_i = ic.run(I1[i], I2[i])
        matrices_individual.append(T_i.matrix)
    M_individual = torch.cat(matrices_individual, dim=0)

    diff = (T_batch.matrix - M_individual).abs().max().item()
    print(f"  Max diff: {diff:.2e}")
    assert diff < 1e-5, f"FAIL: {diff:.2e}"
    print("  PASSED\n")


def test_multiscale_batch_consistency():
    """Multiscale: B=4 batched vs individual."""
    print("=== Multiscale batch consistency ===")

    torch.manual_seed(7)
    B, C, H, W = 4, 1, 64, 64
    dtype = torch.float32

    I1 = torch.randn(B, C, H, W, dtype=dtype)
    I2 = torch.roll(I1, shifts=(1, 2), dims=(2, 3))

    ic = InverseCompositional(
        transform_type='translation', gradient_method='farid5',
        error_function='l2', max_iter=15, dtype=dtype,
    )
    pyr = GaussianPyramid(eta=0.5, sigma_0=0.8, N_scales=3, dtype=dtype)
    ms = MultiscaleIC(ic, pyr, first_scale=0, grayscale=False, dtype=dtype)

    T_batch = ms.run(I1, I2)
    assert T_batch.batch_size == B

    matrices_individual = []
    for i in range(B):
        T_i = ms.run(I1[i], I2[i])
        matrices_individual.append(T_i.matrix)
    M_individual = torch.cat(matrices_individual, dim=0)

    diff = (T_batch.matrix - M_individual).abs().max().item()
    print(f"  Max diff: {diff:.2e}")
    assert diff < 1e-4, f"FAIL: {diff:.2e}"
    print("  PASSED\n")


def test_dtype_propagation():
    """dtype flows correctly through the whole pipeline."""
    print("=== dtype propagation ===")

    for dt in [torch.float32, torch.float64]:
        T = PlanarTransform('homography', batch_size=2, dtype=dt)
        assert T.dtype == dt

        img = torch.randn(2, 1, 16, 16, dtype=dt)
        assert T.warp(img).dtype == dt

        grad = Gradients('farid5', C=1, device='cpu', dtype=dt)
        dx, dy = grad(img)
        assert dx.dtype == dt

        pyr = GaussianPyramid(eta=0.5, sigma_0=0.8, N_scales=3, dtype=dt)
        for lev in pyr(img):
            assert lev.dtype == dt

    print("  PASSED\n")


def test_multiscale_rgb():
    """RGB + grayscale conversion, affinity + Charbonnier."""
    print("=== Multiscale RGB + grayscale ===")

    B, C, H, W = 2, 3, 48, 48
    dtype = torch.float32

    I1 = torch.randn(B, C, H, W, dtype=dtype)
    I2 = I1 + 0.01 * torch.randn_like(I1)

    ic = InverseCompositional(
        transform_type='affinity', gradient_method='farid5',
        error_function='charbonnier', max_iter=10, dtype=dtype,
    )
    pyr = GaussianPyramid(eta=0.5, sigma_0=0.8, N_scales=3, dtype=dtype)
    ms = MultiscaleIC(ic, pyr, first_scale=0, grayscale=True, dtype=dtype)

    T = ms.run(I1, I2)
    assert T.batch_size == B
    assert T.matrix.shape == (B, 3, 3)
    print(f"  Affinity result: {T}")
    print("  PASSED\n")


def bench_time_per_pair_vs_batch_size():
    """Benchmark: time per pair as a function of batch size."""
    import time
    import numpy as np

    print("=== Benchmark: time per pair vs batch size ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    H, W, C = 64, 64, 1
    dtype = torch.float32
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Trim batch sizes that would OOM on small GPUs
    if device.type == 'cuda':
        free_mem = torch.cuda.get_device_properties(0).total_memory
        # rough estimate: ~2MB per pair for 64x64 with intermediates
        max_batch = max(1, int(free_mem * 0.5 / (2 * 1024 * 1024)))
        batch_sizes = [b for b in batch_sizes if b <= max_batch]

    n_warmup = 2
    n_runs = 5

    ic = InverseCompositional(
        transform_type='homography', gradient_method='farid5',
        error_function='lorentzian', max_iter=15, epsilon=1e-4, dtype=dtype,
    )

    results = {}

    for B in batch_sizes:
        torch.manual_seed(0)
        I1 = torch.randn(B, C, H, W, dtype=dtype, device=device)
        I2 = torch.roll(I1, shifts=(2, 3), dims=(2, 3))
        I2 = I2 + 0.02 * torch.randn_like(I2)

        # warmup
        for _ in range(n_warmup):
            _ = ic.run(I1, I2)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = ic.run(I1, I2)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        total_ms = np.median(times) * 1000
        per_pair_ms = total_ms / B
        results[B] = {'total_ms': total_ms, 'per_pair_ms': per_pair_ms}

        print(f"  B={B:>4d}:  total={total_ms:>8.2f} ms  |  per_pair={per_pair_ms:>8.4f} ms")

    # Compute speedup relative to B=1
    if 1 in results:
        baseline = results[1]['per_pair_ms']
        print(f"\n  Speedup vs B=1 ({baseline:.4f} ms/pair):")
        for B, r in results.items():
            speedup = baseline / r['per_pair_ms']
            print(f"    B={B:>4d}:  {speedup:>6.1f}x")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        bs = sorted(results.keys())
        totals = [results[b]['total_ms'] for b in bs]
        per_pair = [results[b]['per_pair_ms'] for b in bs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(bs, totals, 'b-o', lw=2, ms=6)
        ax1.set_xlabel('Batch size')
        ax1.set_ylabel('Total time (ms)')
        ax1.set_title(f'Total time vs batch size ({H}×{W}, {device})')
        ax1.set_xscale('log', base=2)
        ax1.grid(True, alpha=0.3)

        ax2.plot(bs, per_pair, 'r-s', lw=2, ms=6)
        ax2.set_xlabel('Batch size')
        ax2.set_ylabel('Time per pair (ms)')
        ax2.set_title(f'Time per pair vs batch size ({H}×{W}, {device})')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        # annotate speedup
        if 1 in results:
            for b in bs:
                sp = baseline / results[b]['per_pair_ms']
                ax2.annotate(f'{sp:.1f}x', (b, results[b]['per_pair_ms']),
                             textcoords='offset points', xytext=(0, 10),
                             ha='center', fontsize=8, color='darkred')

        plt.tight_layout()
        import os
        os.makedirs('code_tests', exist_ok=True)
        plt.savefig('code_tests/batch_size_benchmark.png', dpi=150)
        plt.close()
        print("\n  Saved: code_tests/batch_size_benchmark.png")
    except ImportError:
        pass

    print("  DONE\n")
    return results


if __name__ == '__main__':
    test_batch_transform_ops()
    test_dtype_propagation()
    test_single_scale_batch_consistency()
    test_single_scale_batch_consistency_robust()
    test_multiscale_batch_consistency()
    test_multiscale_rgb()

    print("=" * 50)
    print("ALL BATCH CONSISTENCY TESTS PASSED")
    print("=" * 50)

    bench_time_per_pair_vs_batch_size()
    print("ALL BATCH CONSISTENCY TESTS PASSED")