"""

--- DISCLAIMER --- 

THIS CODE IS AI WRITTEN
IT IS JUST USED TO COMPARE DIFFERENT IMPLEMENTATIONS

-----------------

Comprehensive Hausdorff distance benchmark for binary masks.

Methods benchmarked (each gracefully skipped if not installed):

OURS:
  1. scipy EDT          — scipy.ndimage.distance_transform_edt, O(H·W)
  2. GPU JFA EDT         — Jump Flooding Algorithm (Rong & Tan, I3D 2006), O(H·W·log(N))
  3. GPU cdist           — torch.cdist brute force, O(N·M)

PUBLISHED / THIRD-PARTY:
  4. scipy KDTree        — scipy.spatial.cKDTree, O(N·log(M))
  5. scipy directed_hd   — scipy.spatial.distance.directed_hausdorff (early-break, Taha & Hanbury TPAMI 2015)
  6. py-hausdorff        — pip install hausdorff (Aziz & Hanbury TPAMI 2015, numba JIT)
  7. MONAI               — pip install monai (DeepMind surface-distance-based, medical imaging standard)
  8. distorch            — pip install distorch (Rony & Kervadec, MIDL 2025, Triton/KeOps GPU)
  9. torchmetrics        — pip install torchmetrics (Lightning AI, scipy-based)

Install all optional deps:
  pip install hausdorff monai distorch torchmetrics
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# OUR IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def hausdorff_edt_scipy(mask1, mask2):
    """Hausdorff via scipy Euclidean Distance Transform. O(H·W)."""
    from scipy.ndimage import distance_transform_edt
    m1 = mask1.bool().cpu().numpy()
    m2 = mask2.bool().cpu().numpy()
    if not m1.any() or not m2.any():
        return float('inf')
    edt_m2 = distance_transform_edt(~m2)
    edt_m1 = distance_transform_edt(~m1)
    return float(max(edt_m2[m1].max(), edt_m1[m2].max()))


def _jfa_edt(mask):
    """Batched EDT via Jump Flooding Algorithm (Rong & Tan, I3D 2006)."""
    B, H, W = mask.shape
    device = mask.device
    INF = float(H + W) * 2.0
    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
    iy = torch.arange(H, device=device)
    ix = torch.arange(W, device=device)
    fg = mask.bool()
    ny = torch.where(fg, yy, torch.full_like(yy, INF))
    nx = torch.where(fg, xx, torch.full_like(xx, INF))
    cur_sq = (yy - ny).square() + (xx - nx).square()
    max_dim = max(H, W)
    step = 1
    while step < max_dim:
        step <<= 1
    step >>= 1
    steps = []
    s = step
    while s >= 1:
        steps.append(s)
        s >>= 1
    steps.append(1)
    for step in steps:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                sy = iy.add(dy).clamp(0, H - 1)
                sx = ix.add(dx).clamp(0, W - 1)
                cand_ny = ny[:, sy][:, :, sx]
                cand_nx = nx[:, sy][:, :, sx]
                cand_sq = (yy - cand_ny).square() + (xx - cand_nx).square()
                better = cand_sq < cur_sq
                ny = torch.where(better, cand_ny, ny)
                nx = torch.where(better, cand_nx, nx)
                cur_sq = torch.where(better, cand_sq, cur_sq)
    return cur_sq.sqrt()


def hausdorff_edt_gpu(mask1, mask2):
    """Hausdorff via JFA EDT on GPU (Rong & Tan, I3D 2006)."""
    m1, m2 = mask1.bool(), mask2.bool()
    if not m1.any() or not m2.any():
        return float('inf')
    stacked = torch.stack([m2, m1], dim=0)
    edts = _jfa_edt(stacked)
    d_1to2 = edts[0][m1].max()
    d_2to1 = edts[1][m2].max()
    return float(torch.max(d_1to2, d_2to1))


def hausdorff_batched_jfa(masks1, masks2):
    """Batched Hausdorff via JFA. All 2B EDTs in one pass."""
    B = masks1.shape[0]
    device = masks1.device
    m1, m2 = masks1.bool(), masks2.bool()
    stacked = torch.stack([m2, m1], dim=1).reshape(2 * B, *m1.shape[1:])
    edts = _jfa_edt(stacked).reshape(B, 2, *m1.shape[1:])
    results = torch.zeros(B, device=device)
    for b in range(B):
        if not m1[b].any() or not m2[b].any():
            results[b] = float('inf')
            continue
        results[b] = torch.max(edts[b, 0][m1[b]].max(), edts[b, 1][m2[b]].max())
    return results


def hausdorff_gpu_cdist(mask1, mask2):
    """Hausdorff via full pairwise distance matrix. O(N·M)."""
    c1 = torch.nonzero(mask1, as_tuple=False).float()
    c2 = torch.nonzero(mask2, as_tuple=False).float()
    if c1.shape[0] == 0 or c2.shape[0] == 0:
        return float('inf')
    D = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)).squeeze(0)
    return float(torch.max(D.min(1).values.max(), D.min(0).values.max()))


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def bench(fn, n_warmup=3, n_repeat=20, sync_cuda=True):
    """Time a function, return (result, median_ms)."""
    device_is_cuda = torch.cuda.is_available()
    for _ in range(n_warmup):
        result = fn()
    if device_is_cuda and sync_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        if device_is_cuda and sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if device_is_cuda and sync_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times)) * 1000


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Load MNIST and upscale ──────────────────────────────────────────────
    from torchvision import datasets, transforms

    mnist = datasets.MNIST(root='/tmp/mnist', train=True, download=True,
                           transform=transforms.ToTensor())

    TARGET_SIZE = 256
    THRESHOLD = 0.3

    def mnist_to_mask(idx, size=TARGET_SIZE):
        img = mnist[idx][0]
        img = F.interpolate(img.unsqueeze(0), size=(size, size),
                            mode='bilinear', align_corners=False)
        return (img.squeeze() > THRESHOLD).to(device)

    mask1 = mnist_to_mask(0)
    mask2 = mnist_to_mask(1)
    n1, n2 = mask1.sum().item(), mask2.sum().item()
    print(f"MNIST upscaled to {TARGET_SIZE}x{TARGET_SIZE}, threshold={THRESHOLD}")
    print(f"Mask sizes: {n1:.0f} and {n2:.0f} foreground pixels")
    print(f"{'='*72}\n")

    # numpy versions for CPU methods
    m1_np = mask1.cpu().numpy().astype(bool)
    m2_np = mask2.cpu().numpy().astype(bool)

    # point-set versions for KDTree / py-hausdorff / scipy.directed_hausdorff
    c1_np = np.argwhere(m1_np).astype(np.float64)
    c2_np = np.argwhere(m2_np).astype(np.float64)

    N_REPEAT = 30

    results = {}  # name → (hd_value, time_ms, source_info)

    # ────────────────────────────────────────────────────────────────────────
    # 1. scipy KDTree (baseline)
    # ────────────────────────────────────────────────────────────────────────
    from scipy.spatial import cKDTree

    def fn_kdtree():
        t1 = cKDTree(c1_np)
        t2 = cKDTree(c2_np)
        d1, _ = t2.query(c1_np, workers=-1)
        d2, _ = t1.query(c2_np, workers=-1)
        return max(d1.max(), d2.max())

    hd, t = bench(fn_kdtree, n_repeat=N_REPEAT, sync_cuda=False)
    results['scipy KDTree'] = (hd, t, 'scipy.spatial.cKDTree — O(N log M)')
    print(f"  [scipy KDTree]         HD={hd:>8.4f}  time={t:>8.2f} ms")

    # ────────────────────────────────────────────────────────────────────────
    # 2. scipy EDT (ours)
    # ────────────────────────────────────────────────────────────────────────
    hd, t = bench(lambda: hausdorff_edt_scipy(mask1, mask2),
                  n_repeat=N_REPEAT, sync_cuda=False)
    results['scipy EDT'] = (hd, t, 'scipy.ndimage.distance_transform_edt — O(H·W)')
    print(f"  [scipy EDT]            HD={hd:>8.4f}  time={t:>8.2f} ms")

    # ────────────────────────────────────────────────────────────────────────
    # 3. GPU JFA EDT (ours)
    # ────────────────────────────────────────────────────────────────────────
    hd, t = bench(lambda: hausdorff_edt_gpu(mask1, mask2), n_repeat=N_REPEAT)
    results['GPU JFA EDT'] = (hd, t, 'Jump Flooding (Rong & Tan, I3D 2006) — O(HW log N)')
    print(f"  [GPU JFA EDT]          HD={hd:>8.4f}  time={t:>8.2f} ms")

    # ────────────────────────────────────────────────────────────────────────
    # 4. GPU cdist (ours, brute force)
    # ────────────────────────────────────────────────────────────────────────
    hd, t = bench(lambda: hausdorff_gpu_cdist(mask1, mask2), n_repeat=N_REPEAT)
    results['GPU cdist'] = (hd, t, 'torch.cdist brute-force — O(N·M)')
    print(f"  [GPU cdist]            HD={hd:>8.4f}  time={t:>8.2f} ms")

    # ────────────────────────────────────────────────────────────────────────
    # 5. scipy.spatial.distance.directed_hausdorff (early-break)
    #    Implements: Taha & Hanbury, TPAMI 2015
    # ────────────────────────────────────────────────────────────────────────
    try:
        from scipy.spatial.distance import directed_hausdorff

        def fn_scipy_dh():
            d1 = directed_hausdorff(c1_np, c2_np)[0]
            d2 = directed_hausdorff(c2_np, c1_np)[0]
            return max(d1, d2)

        hd, t = bench(fn_scipy_dh, n_repeat=N_REPEAT, sync_cuda=False)
        results['scipy directed_hd'] = (hd, t,
            'scipy.spatial.distance.directed_hausdorff — early-break (Taha & Hanbury TPAMI 2015)')
        print(f"  [scipy directed_hd]    HD={hd:>8.4f}  time={t:>8.2f} ms")
    except ImportError:
        print("  [scipy directed_hd]    SKIPPED (not available)")

    # ────────────────────────────────────────────────────────────────────────
    # 6. py-hausdorff — pip install hausdorff
    #    Aziz & Hanbury, "Efficient Algorithm for Exact Hausdorff Distance",
    #    TPAMI 2015. Numba JIT implementation.
    # ────────────────────────────────────────────────────────────────────────
    try:
        from hausdorff import hausdorff_distance as hd_aziz

        # warmup numba JIT
        _ = hd_aziz(c1_np[:10], c2_np[:10])

        def fn_pyh():
            return hd_aziz(c1_np, c2_np, distance='euclidean')

        hd, t = bench(fn_pyh, n_repeat=N_REPEAT, sync_cuda=False)
        results['py-hausdorff'] = (hd, t,
            'hausdorff (Aziz & Hanbury TPAMI 2015) — numba JIT, early-break')
        print(f"  [py-hausdorff]         HD={hd:>8.4f}  time={t:>8.2f} ms")
    except ImportError:
        print("  [py-hausdorff]         SKIPPED — pip install hausdorff")

    # ────────────────────────────────────────────────────────────────────────
    # 7. MONAI — pip install monai
    #    Project-MONAI, based on DeepMind's surface-distance.
    #    Internally: edge extraction + scipy EDT + KDTree.
    # ────────────────────────────────────────────────────────────────────────
    try:
        from monai.metrics import compute_hausdorff_distance as monai_hd

        # MONAI expects (B, C, H, W) one-hot format
        pred_monai = mask1.unsqueeze(0).unsqueeze(0).float()
        gt_monai = mask2.unsqueeze(0).unsqueeze(0).float()

        def fn_monai():
            return monai_hd(pred_monai, gt_monai, include_background=True).item()

        hd, t = bench(fn_monai, n_repeat=N_REPEAT, sync_cuda=False)
        results['MONAI'] = (hd, t,
            'monai.metrics.compute_hausdorff_distance — EDT + surface extraction (DeepMind)')
        print(f"  [MONAI]                HD={hd:>8.4f}  time={t:>8.2f} ms")
    except ImportError:
        print("  [MONAI]                SKIPPED — pip install monai")

    # ────────────────────────────────────────────────────────────────────────
    # 8. distorch — pip install distorch
    #    Rony & Kervadec, MIDL 2025. Triton/KeOps GPU kernels.
    # ────────────────────────────────────────────────────────────────────────
    try:
        import distorch

        # distorch.boundary_metrics expects 2D or (B, H, W) bool tensors on GPU
        def fn_distorch():
            m = distorch.boundary_metrics(mask1, mask2)
            return m.Hausdorff.item()

        hd, t = bench(fn_distorch, n_repeat=N_REPEAT)
        results['distorch'] = (hd, t,
            'distorch (Rony & Kervadec, MIDL 2025) — Triton/KeOps GPU')
        print(f"  [distorch]             HD={hd:>8.4f}  time={t:>8.2f} ms")
    except ImportError:
        print("  [distorch]             SKIPPED — pip install distorch")
    except Exception as e:
        print(f"  [distorch]             ERROR: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # 9. torchmetrics — pip install torchmetrics
    #    Lightning AI. Internally uses scipy EDT.
    # ────────────────────────────────────────────────────────────────────────
    try:
        from torchmetrics.functional.segmentation import hausdorff_distance as tm_hd

        # torchmetrics expects (B, C, H, W) one-hot bool tensors
        pred_tm = mask1.unsqueeze(0).unsqueeze(0)
        gt_tm = mask2.unsqueeze(0).unsqueeze(0)

        def fn_torchmetrics():
            return tm_hd(pred_tm, gt_tm, num_classes=1,
                        include_background=True).item()

        hd, t = bench(fn_torchmetrics, n_repeat=N_REPEAT, sync_cuda=False)
        results['torchmetrics'] = (hd, t,
            'torchmetrics (Lightning AI) — scipy EDT under the hood')
        print(f"  [torchmetrics]         HD={hd:>8.4f}  time={t:>8.2f} ms")
    except ImportError:
        print("  [torchmetrics]         SKIPPED — pip install torchmetrics")
    except Exception as e:
        print(f"  [torchmetrics]         ERROR: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════════════
    baseline_t = results['scipy KDTree'][1]
    baseline_hd = results['scipy KDTree'][0]

    print(f"\n{'='*72}")
    print(f"  SUMMARY — {TARGET_SIZE}x{TARGET_SIZE} MNIST masks, "
          f"{n1:.0f}/{n2:.0f} fg pixels")
    print(f"{'='*72}")
    print(f"  {'Method':<22} {'HD':>8} {'Time(ms)':>10} {'vs KDTree':>10}  "
          f"{'|Δ HD|':>8}")
    print(f"  {'─'*66}")

    for name in ['scipy KDTree', 'scipy EDT', 'scipy directed_hd',
                 'py-hausdorff', 'MONAI', 'torchmetrics',
                 'distorch', 'GPU JFA EDT', 'GPU cdist']:
        if name not in results:
            continue
        hd_val, t_ms, info = results[name]
        speedup = baseline_t / t_ms
        delta = abs(hd_val - baseline_hd)
        marker = '★' if speedup >= 1.5 else ('✓' if speedup >= 0.9 else ' ')
        print(f"  {marker} {name:<20} {hd_val:>8.4f} {t_ms:>10.2f} "
              f"{speedup:>9.1f}x  {delta:>8.4f}")

    print(f"  {'─'*66}")

    # ── Batched JFA benchmark ───────────────────────────────────────────────
    B = 64
    masks1_b = torch.stack([mnist_to_mask(i) for i in range(B)])
    masks2_b = torch.stack([mnist_to_mask(i + B) for i in range(B)])

    hds, t_batched = bench(lambda: hausdorff_batched_jfa(masks1_b, masks2_b),
                           n_warmup=3, n_repeat=5)
    per_pair = t_batched / B

    print(f"\n  Batched JFA (B={B}):  total={t_batched:.2f} ms  "
          f"per_pair={per_pair:.2f} ms  ({baseline_t/per_pair:.1f}x vs KDTree)")
    print(f"  HD range: [{hds.min():.2f}, {hds.max():.2f}]")

    # ── Batched distorch if available ───────────────────────────────────────
    if 'distorch' in results:
        try:
            import distorch

            def fn_distorch_batch():
                m = distorch.boundary_metrics(masks1_b, masks2_b)
                return m.Hausdorff

            hds_dt, t_dt = bench(fn_distorch_batch, n_warmup=3, n_repeat=5)
            per_pair_dt = t_dt / B
            print(f"  Batched distorch (B={B}):  total={t_dt:.2f} ms  "
                  f"per_pair={per_pair_dt:.2f} ms  ({baseline_t/per_pair_dt:.1f}x vs KDTree)")
        except Exception as e:
            print(f"  Batched distorch: ERROR — {e}")

    # ── References ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  REFERENCES")
    print(f"{'='*72}")
    for name, (_, _, info) in sorted(results.items()):
        print(f"  • {name}: {info}")

    # ── Install suggestions ─────────────────────────────────────────────────
    all_methods = {'scipy directed_hd', 'py-hausdorff', 'MONAI',
                   'distorch', 'torchmetrics'}
    missing = all_methods - set(results.keys())
    if missing:
        print(f"\n  To enable all methods, install:")
        print(f"    pip install hausdorff monai distorch torchmetrics")

    # ── Visualisation ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os; os.makedirs('code_tests', exist_ok=True)

        names = [n for n in ['scipy KDTree', 'scipy EDT', 'scipy directed_hd',
                             'py-hausdorff', 'MONAI', 'torchmetrics',
                             'distorch', 'GPU JFA EDT', 'GPU cdist']
                 if n in results]
        times = [results[n][1] for n in names]
        colors = ['#2196F3' if 'scipy' in n or n in ('MONAI', 'torchmetrics', 'py-hausdorff')
                  else '#FF5722' for n in names]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(names, times, color=colors, edgecolor='white', height=0.6)

        # Annotate speedup
        for bar, name in zip(bars, names):
            t = results[name][1]
            sp = baseline_t / t
            label = f'{t:.2f} ms ({sp:.1f}x)'
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=9)

        ax.set_xlabel('Time (ms)')
        ax.set_title(f'Hausdorff Distance Benchmark — {TARGET_SIZE}×{TARGET_SIZE} '
                     f'MNIST masks ({n1:.0f}/{n2:.0f} px)')
        ax.axvline(x=baseline_t, color='gray', linestyle='--', alpha=0.5,
                   label=f'KDTree baseline ({baseline_t:.1f} ms)')
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('code_tests/hausdorff_benchmark_all.png', dpi=150,
                    bbox_inches='tight')
        plt.close()
        print("\n  Saved: code_tests/hausdorff_benchmark_all.png")
    except ImportError:
        pass