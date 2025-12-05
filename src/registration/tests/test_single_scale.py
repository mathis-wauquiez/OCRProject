from ..single_scale import *

# ═══════════════════════════════════════════════════════════════════════════════
# TEST WITH PROFILING AND GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from contextlib import contextmanager
    import time
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    import matplotlib.pyplot as plt
    import os

    def _to_numpy(img):
        """Convert BCHW → HWC and clamp for plotting."""
        if img.dim() == 4:
            img = img[0]
        return img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    def save_eval_plot(
        I1, I2, T_hat, ic, save_dir, tag="default"
    ):
        """
        I1, I2: tensors [1, C, H, W] in [0,1]
        T_hat: estimated transformation
        ic: image comparator providing error_fn_class()
        """

        os.makedirs(save_dir, exist_ok=True)

        # ---- Compute warped image ----
        with torch.no_grad():
            I2_warped = T_hat.warp(I2)  # returns only the warped image

        # get valid mask from transform geometry
        H, W = I2.shape[2:]
        valid_mask = T_hat.visibility_mask(H, W).float().unsqueeze(0).unsqueeze(0)

        # ---- Compute diff_masked ----
        diff = I1 - I2_warped
        diff_masked = diff * valid_mask

        # ---- Compute scalar-per-pixel error ----
        err = ic.error_fn_class().rho_prime(diff_masked)  # [1,1,H,W]

        # ---- Convert tensors to numpy for plotting ----
        def to_np(x):
            x = x.detach().cpu()
            if x.shape[1] == 1:
                return x[0, 0].numpy()
            else:
                return x[0].permute(1, 2, 0).numpy()

        I1_np         = to_np(I1)
        I2_np         = to_np(I2)
        I2w_np        = to_np(I2_warped)
        mask_np       = to_np(valid_mask)
        diff_m_np     = to_np(diff_masked)
        err_np        = to_np(err)

        # ---- Visualization ----
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        panels = [
            ("I1 (reference)", I1_np),
            ("I2 (source)", I2_np),
            ("I2 warped", I2w_np),
            ("valid_mask", mask_np),
            ("diff_masked", diff_m_np),
            ("error_fn_class(diff_masked)", err_np),
        ]

        for ax, (title, img) in zip(axes.ravel(), panels):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")

        out_path = os.path.join(save_dir, f"{tag}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        return out_path



    
    class Timer:
        """Simple profiling timer."""
        def __init__(self):
            self.times = {}
            self.counts = {}
        
        @contextmanager
        def __call__(self, name):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            yield
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start
            if name not in self.times:
                self.times[name] = 0.0
                self.counts[name] = 0
            self.times[name] += elapsed
            self.counts[name] += 1
        
        def reset(self):
            self.times = {}
            self.counts = {}
        
        def get_grouped(self):
            """Group times by prefix (precompute/iter/etc)."""
            groups = {}
            for name, t in self.times.items():
                prefix = name.split('/')[0]
                if prefix not in groups:
                    groups[prefix] = 0.0
                groups[prefix] += t
            return groups
        
        def summary(self):
            total = sum(self.times.values())
            print(f"\n{'='*65}")
            print(f"{'PROFILING SUMMARY':^65}")
            print(f"{'='*65}")
            print(f"{'Operation':<35} {'Time (ms)':>10} {'Calls':>8} {'%':>8}")
            print(f"{'-'*65}")
            for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
                pct = 100 * t / total if total > 0 else 0
                print(f"{name:<35} {t*1000:>10.3f} {self.counts[name]:>8} {pct:>7.1f}%")
            print(f"{'-'*65}")
            print(f"{'TOTAL':<35} {total*1000:>10.3f}")
            print(f"{'='*65}\n")
    
    def load_test_image(size=128):
        try:
            from sklearn.datasets import load_sample_image
            img = load_sample_image('china.jpg').copy()
            h, w = img.shape[:2]
            s = min(h, w)
            img = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
        except:
            img = torch.zeros(1, 3, size, size)
            for i in range(0, size, 16):
                for j in range(0, size, 16):
                    if ((i // 16) + (j // 16)) % 2 == 0:
                        img[0, :, i:i+16, j:j+16] = torch.tensor([0.9, 0.3, 0.3])[:, None, None]
                    else:
                        img[0, :, i:i+16, j:j+16] = torch.tensor([0.3, 0.5, 0.9])[:, None, None]
            return img

    def run_profiled(I1, I2, ic, timer):
        """Run IC algorithm with detailed profiling."""
        if I1.dim() == 3: I1 = I1.unsqueeze(0)
        if I2.dim() == 3: I2 = I2.unsqueeze(0)
        
        B, C, H, W = I1.shape
        device, dtype = I1.device, I1.dtype
        
        T = PlanarTransform(ic.transform_type, device=device, dtype=dtype)
        
        # Precomputations
        with timer('precompute/gradient_module'):
            grad = Gradients(ic.gradient_method, C, device)
        
        with timer('precompute/gradient_I1'):
            dI1_dx, dI1_dy = grad(I1)
        
        with timer('precompute/prefilter'):
            I1_tilde = ic._prefilter(I1, grad)
            I2_tilde = ic._prefilter(I2, grad)
        
        with timer('precompute/jacobian'):
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing='ij'
            )
            J = PlanarTransform(ic.transform_type).jacobian(xx, yy)
        
        with timer('precompute/G'):
            grad_I = torch.stack([dI1_dx, dI1_dy], dim=-1)
            G = torch.einsum('bchwd,hwdp->bchwp', grad_I, J)
        
        with timer('precompute/GTG'):
            GTG = torch.einsum('bchwp,bchwq->bhwpq', G, G)
        
        # Iterations
        error_fn = ic.error_fn_class(ic.lambda_init)
        n_iters = 0
        
        for j in range(ic.max_iter):
            n_iters += 1
            
            if ic.robust:
                error_fn.lam = max(ic.lambda_decay ** j * ic.lambda_init, ic.lambda_min)
            
            with timer('iter/visibility_mask'):
                mask = T.visibility_mask(H, W, delta=ic.delta).float()
            
            with timer('iter/warp'):
                I2_warped = T.warp(I2_tilde)
            
            with timer('iter/difference'):
                DI = I2_warped - I1_tilde
                DI_norm2 = (DI ** 2).sum(dim=1)
            
            with timer('iter/weights'):
                weight = error_fn.rho_prime(DI_norm2) * mask
            
            with timer('iter/compute_b'):
                b = torch.einsum('bhw,bchwp,bchw->bp', weight, G, DI)
            
            with timer('iter/compute_H'):
                H_mat = torch.einsum('bhw,bhwpq->bpq', weight, GTG)
            
            with timer('iter/solve'):
                delta_p = torch.linalg.solve(H_mat, b)
            
            if delta_p.norm(dim=-1).max() <= ic.epsilon:
                break
            
            with timer('iter/update_transform'):
                T_delta = PlanarTransform(ic.transform_type, params=delta_p[0],
                                          device=device, dtype=dtype)
                T = T @ T_delta.inv
        
        return T, n_iters

    # ───────────────────────────────────────────────────────────────────────
    # Benchmark across resolutions
    # ───────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*65)
    print(" INVERSE COMPOSITIONAL ALGORITHM - BENCHMARK ")
    print("="*65)
    
    sizes = [64, 128, 192, 256, 320, 384, 448, 512]
    n_runs = 5  # Average over multiple runs

    # sizes = [64, 128, 256]
    # n_runs = 2  # Average over multiple runs

    # Storage for results
    results = {
        'sizes': sizes,
        'total_time': [],
        'precompute_time': [],
        'iter_time': [],
        'epe': [],
        'n_iters': [],
        'time_breakdown': [],
    }
    
    for size in sizes:
        print(f"\nBenchmarking {size}x{size}...")
        
        I2 = load_test_image(size).to(device)
        T_gt = PlanarTransform('homography',
                               params=[0.05, 0.02, 8.0, -0.02, 0.04, 5.0, 0.0001, -0.0001])
        I1 = T_gt.warp(I2)
        
        ic = InverseCompositional(
            transform_type='homography',
            gradient_method='farid5',
            error_function='lorentzian',
            delta=5,
            max_iter=5000
        )
        
        run_times = []
        run_precompute = []
        run_iter = []
        run_epe = []
        run_n_iters = []
        breakdown_accum = {}
        
        for run in range(n_runs):
            timer = Timer()
            T_est, n_iters = run_profiled(I1, I2, ic, timer)
            
            # Compute EPE
            H, W = size, size
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
            yy = yy.to(I1.device) ; xx = xx.to(I1.device)
            xx_gt, yy_gt = T_gt.transform_points(xx, yy)
            xx_est, yy_est = T_est.transform_points(xx, yy)
            epe = torch.sqrt((xx_gt - xx_est)**2 + (yy_gt - yy_est)**2).mean().item()
            
            # Gather times
            groups = timer.get_grouped()
            total = sum(timer.times.values()) * 1000
            precompute = groups.get('precompute', 0) * 1000
            iter_time = groups.get('iter', 0) * 1000
            
            run_times.append(total)
            run_precompute.append(precompute)
            run_iter.append(iter_time)
            run_epe.append(epe)
            run_n_iters.append(n_iters)
            
            # Accumulate breakdown
            for name, t in timer.times.items():
                if name not in breakdown_accum:
                    breakdown_accum[name] = []
                breakdown_accum[name].append(t * 1000)
        
        results['total_time'].append(np.mean(run_times))
        results['precompute_time'].append(np.mean(run_precompute))
        results['iter_time'].append(np.mean(run_iter))
        results['epe'].append(np.mean(run_epe))
        results['n_iters'].append(np.mean(run_n_iters))
        results['time_breakdown'].append({k: np.mean(v) for k, v in breakdown_accum.items()})
        
        print(f"  Time: {np.mean(run_times):.2f} ± {np.std(run_times):.2f} ms")
        print(f"  EPE:  {np.mean(run_epe):.6f} px")
        print(f"  Iters: {np.mean(run_n_iters):.1f}")


    # ───────────────────────────────────────────────────────────────────────
    # Create visualization
    # ───────────────────────────────────────────────────────────────────────
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Total time vs resolution
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(sizes, results['total_time'], 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Image Size (pixels)', fontsize=12)
    ax1.set_ylabel('Total Time (ms)', fontsize=12)
    ax1.set_title('Total Execution Time vs Resolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sizes)
    
    # 2. Precompute vs Iteration time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(sizes, results['precompute_time'], 'g-o', linewidth=2, markersize=8, label='Precompute')
    ax2.plot(sizes, results['iter_time'], 'r-s', linewidth=2, markersize=8, label='Iterations')
    ax2.set_xlabel('Image Size (pixels)', fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Precompute vs Iteration Time', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sizes)
    
    # 3. EPE vs resolution
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(sizes, results['epe'], 'm-o', linewidth=2, markersize=8)
    ax3.set_xlabel('Image Size (pixels)', fontsize=12)
    ax3.set_ylabel('Mean EPE (pixels)', fontsize=12)
    ax3.set_title('End-Point Error vs Resolution', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(sizes)
    ax3.set_yscale('log')
    
    # 4. Time per pixel (complexity analysis)
    ax4 = fig.add_subplot(2, 3, 4)
    pixels = [s*s for s in sizes]
    time_per_pixel = [t / p * 1000 for t, p in zip(results['total_time'], pixels)]  # µs per pixel
    ax4.plot(sizes, time_per_pixel, 'c-o', linewidth=2, markersize=8)
    ax4.set_xlabel('Image Size (pixels)', fontsize=12)
    ax4.set_ylabel('Time per Pixel (µs)', fontsize=12)
    ax4.set_title('Time Complexity per Pixel', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(sizes)
    
    # 5. Stacked bar chart of time breakdown
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Group operations for cleaner visualization
    op_groups = {
        'Gradient': ['precompute/gradient_module', 'precompute/gradient_I1'],
        'Prefilter': ['precompute/prefilter'],
        'Jacobian/G/GTG': ['precompute/jacobian', 'precompute/G', 'precompute/GTG'],
        'Warp': ['iter/warp'],
        'Mask': ['iter/visibility_mask'],
        'H & b': ['iter/compute_H', 'iter/compute_b', 'iter/difference', 'iter/weights'],
        'Solve': ['iter/solve'],
        'Update': ['iter/update_transform'],
    }
    
    bar_data = {group: [] for group in op_groups}
    for breakdown in results['time_breakdown']:
        for group, ops in op_groups.items():
            total = sum(breakdown.get(op, 0) for op in ops)
            bar_data[group].append(total)
    
    bottom = np.zeros(len(sizes))
    colors = plt.cm.tab10(np.linspace(0, 1, len(op_groups)))
    
    for (group, values), color in zip(bar_data.items(), colors):
        ax5.bar(range(len(sizes)), values, bottom=bottom, label=group, color=color)
        bottom += np.array(values)
    
    ax5.set_xticks(range(len(sizes)))
    ax5.set_xticklabels(sizes)
    ax5.set_xlabel('Image Size (pixels)', fontsize=12)
    ax5.set_ylabel('Time (ms)', fontsize=12)
    ax5.set_title('Time Breakdown by Operation', fontsize=14)
    ax5.legend(loc='upper left', fontsize=8)
    
    # 6. Number of iterations
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(range(len(sizes)), results['n_iters'], color='orange', edgecolor='black')
    ax6.set_xticks(range(len(sizes)))
    ax6.set_xticklabels(sizes)
    ax6.set_xlabel('Image Size (pixels)', fontsize=12)
    ax6.set_ylabel('Number of Iterations', fontsize=12)
    ax6.set_title('Convergence: Iterations to Converge', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('code_tests/ic_benchmark_resolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: ic_benchmark_resolution.png")

    # ───────────────────────────────────────────────────────────────────────
    # Compare error functions across resolutions
    # ───────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*65)
    print(" COMPARE ERROR FUNCTIONS ACROSS RESOLUTIONS ")
    print("="*65)
    
    test_sizes = [32, 64, 128, 256, 512]
    error_functions = ['l2', 'lorentzian', 'geman_mcclure', 'charbonnier']
    
    err_results = {ef: {'time': [], 'epe': []} for ef in error_functions}
    
    for size in test_sizes:
        print(f"\nSize: {size}x{size}")
        I2 = load_test_image(size).to(device)
        T_gt = PlanarTransform('homography',
                               params=[0.05, 0.02, 8.0, -0.02, 0.04, 5.0, 0.0001, -0.0001])
        I1 = T_gt.warp(I2)
        
        H, W = size, size
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
        yy = yy.to(I1.device) ; xx = xx.to(I1.device)
        xx_gt, yy_gt = T_gt.transform_points(xx, yy)
        
        for err_fn in error_functions:
            ic = InverseCompositional(
                transform_type='homography',
                gradient_method='farid5',
                error_function=err_fn,
                delta=5,
                max_iter=30
            )
            
            # Average over 3 runs
            times, epes = [], []
            for _ in range(3):
                start = time.perf_counter()
                T_est = ic.run(I1, I2)
                elapsed = (time.perf_counter() - start) * 1000
                
                xx_est, yy_est = T_est.transform_points(xx, yy)
                epe = torch.sqrt((xx_gt - xx_est)**2 + (yy_gt - yy_est)**2).mean().item()
                
                times.append(elapsed)
                epes.append(epe)
            
            err_results[err_fn]['time'].append(np.mean(times))
            err_results[err_fn]['epe'].append(np.mean(epes))
            print(f"  {err_fn:<15}: {np.mean(times):>8.2f} ms, EPE={np.mean(epes):.6f}")
    
    # Plot error function comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(test_sizes))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (err_fn, color) in enumerate(zip(error_functions, colors)):
        axes[0].bar(x + i*width, err_results[err_fn]['time'], width, label=err_fn, color=color)
        axes[1].bar(x + i*width, err_results[err_fn]['epe'], width, label=err_fn, color=color)
    
    axes[0].set_xlabel('Image Size', fontsize=12)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Execution Time by Error Function', fontsize=14)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels([f'{s}x{s}' for s in test_sizes])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].set_xlabel('Image Size', fontsize=12)
    axes[1].set_ylabel('Mean EPE (pixels)', fontsize=12)
    axes[1].set_title('Accuracy by Error Function', fontsize=14)
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels([f'{s}x{s}' for s in test_sizes])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('code_tests/ic_benchmark_error_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: ic_benchmark_error_functions.png")

    # ───────────────────────────────────────────────────────────────────────
    # Compare gradient methods across resolutions
    # ───────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*65)
    print(" COMPARE GRADIENT METHODS ACROSS RESOLUTIONS ")
    print("="*65)
    
    gradient_methods = ['central', 'farid3', 'farid5']
    grad_results = {gm: {'time': [], 'epe': []} for gm in gradient_methods}
    
    for size in test_sizes:
        print(f"\nSize: {size}x{size}")
        I2 = load_test_image(size).to(device)
        T_gt = PlanarTransform('homography',
                               params=[0.05, 0.02, 8.0, -0.02, 0.04, 5.0, 0.0001, -0.0001])
        I1 = T_gt.warp(I2)
        
        H, W = size, size
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
        yy = yy.to(I1.device) ; xx = xx.to(I1.device)

        xx_gt, yy_gt = T_gt.transform_points(xx, yy)
        
        for grad_method in gradient_methods:
            ic = InverseCompositional(
                transform_type='homography',
                gradient_method=grad_method,
                error_function='lorentzian',
                delta=5,
                max_iter=30
            )
            
            times, epes = [], []
            for _ in range(3):
                start = time.perf_counter()
                T_est = ic.run(I1, I2)
                elapsed = (time.perf_counter() - start) * 1000
                
                xx_est, yy_est = T_est.transform_points(xx, yy)
                epe = torch.sqrt((xx_gt - xx_est)**2 + (yy_gt - yy_est)**2).mean().item()
                
                times.append(elapsed)
                epes.append(epe)
            
            grad_results[grad_method]['time'].append(np.mean(times))
            grad_results[grad_method]['epe'].append(np.mean(epes))
            print(f"  {grad_method:<15}: {np.mean(times):>8.2f} ms, EPE={np.mean(epes):.6f}")
    

            out_path = save_eval_plot(
                I1=I1,
                I2=I2,
                T_hat=T_est,
                ic=ic,
                save_dir="code_tests/single_scale_viz",
                tag=f"{size}_{grad_method}"
            )
            print("Saved:", out_path)





    # Plot gradient method comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (grad_method, color) in enumerate(zip(gradient_methods, colors)):
        axes[0].bar(x + i*width, grad_results[grad_method]['time'], width, label=grad_method, color=color)
        axes[1].bar(x + i*width, grad_results[grad_method]['epe'], width, label=grad_method, color=color)
    
    axes[0].set_xlabel('Image Size', fontsize=12)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Execution Time by Gradient Method', fontsize=14)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([f'{s}x{s}' for s in test_sizes])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].set_xlabel('Image Size', fontsize=12)
    axes[1].set_ylabel('Mean EPE (pixels)', fontsize=12)
    axes[1].set_title('Accuracy by Gradient Method', fontsize=14)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f'{s}x{s}' for s in test_sizes])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('code_tests/ic_benchmark_gradient_methods.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: ic_benchmark_gradient_methods.png")

    # ───────────────────────────────────────────────────────────────────────
    # Scaling analysis: O(n) complexity check
    # ───────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*65)
    print(" COMPLEXITY ANALYSIS ")
    print("="*65)
    
    pixels = np.array([s*s for s in sizes])
    times = np.array(results['total_time'])
    
    # Fit linear and quadratic models
    from numpy.polynomial import polynomial as P
    
    # Linear fit: t = a * n + b
    linear_coef = np.polyfit(pixels, times, 1)
    linear_fit = np.polyval(linear_coef, pixels)
    
    # Quadratic fit: t = a * n^2 + b * n + c  
    quad_coef = np.polyfit(pixels, times, 2)
    quad_fit = np.polyval(quad_coef, pixels)
    
    # R² scores
    ss_tot = np.sum((times - np.mean(times))**2)
    r2_linear = 1 - np.sum((times - linear_fit)**2) / ss_tot
    r2_quad = 1 - np.sum((times - quad_fit)**2) / ss_tot
    
    print(f"Linear fit R²:    {r2_linear:.4f}")
    print(f"Quadratic fit R²: {r2_quad:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(pixels, times, s=100, c='blue', label='Measured', zorder=3)
    
    # Extended x for smoother curves
    x_smooth = np.linspace(pixels.min(), pixels.max(), 100)
    ax.plot(x_smooth, np.polyval(linear_coef, x_smooth), 'g--', 
            linewidth=2, label=f'Linear fit (R²={r2_linear:.3f})')
    ax.plot(x_smooth, np.polyval(quad_coef, x_smooth), 'r:', 
            linewidth=2, label=f'Quadratic fit (R²={r2_quad:.3f})')
    
    ax.set_xlabel('Number of Pixels', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Time Complexity Analysis', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add size labels
    for s, p, t in zip(sizes, pixels, times):
        ax.annotate(f'{s}x{s}', (p, t), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('code_tests/ic_benchmark_complexity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: ic_benchmark_complexity.png")

    # ───────────────────────────────────────────────────────────────────────
    # Summary table
    # ───────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*65)
    print(" SUMMARY TABLE ")
    print("="*65)
    print(f"{'Size':>8} {'Pixels':>10} {'Time (ms)':>12} {'EPE (px)':>12} {'Iters':>8} {'ms/Mpx':>10}")
    print("-"*65)
    for i, size in enumerate(sizes):
        px = size * size
        ms_per_mpx = results['total_time'][i] / px * 1e6
        print(f"{size:>8} {px:>10} {results['total_time'][i]:>12.2f} "
              f"{results['epe'][i]:>12.6f} {results['n_iters'][i]:>8.1f} {ms_per_mpx:>10.2f}")
        

