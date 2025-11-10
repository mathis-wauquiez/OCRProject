import torch
import numpy as np
import matplotlib.pyplot as plt


class LoopyBeliefPropagation:
    """
    Loopy Belief Propagation with Potts model for grid MRF inference.
    
    Minimizes: E(L) = sum_p unary_cost[p, l_p] + lambda * sum_{p,q} I(l_p != l_q)
    where I(.) is the indicator function (Potts model).
    
    Examples:
        >>> # With your own unary costs (H, W, num_labels)
        >>> lbp = LoopyBeliefPropagation(lambda_smooth=10.0)
        >>> labels, energies = lbp.fit(unary_cost)
        
        >>> # Stereo matching
        >>> unary = compute_stereo_cost(left, right, dmax=16)
        >>> labels, energies = lbp.fit(unary)
    """
    
    def __init__(self, lambda_smooth=1.0, max_iter=50, verbose=True):
        """
        Args:
            lambda_smooth: Weight for smoothness term (higher = smoother)
            max_iter: Number of message passing iterations
            verbose: Print iteration progress
        """
        self.lambda_smooth = lambda_smooth
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, unary_cost):
        """
        Run LBP inference to find MAP labeling.
        
        Args:
            unary_cost: Tensor or array of shape (H, W, num_labels)
                       Cost of assigning each label to each pixel
            
        Returns:
            labels: (H, W) numpy array of optimal labels
            energies: List of total energy at each iteration
        """
        unary_cost = self._to_tensor(unary_cost)
        h, w, num_labels = unary_cost.shape
        
        if self.verbose:
            print(f"LBP: {h}×{w} grid, {num_labels} labels, λ={self.lambda_smooth}")
        
        # Initialize all messages to zero
        messages = {d: torch.zeros(h, w, num_labels, device=self.device)
                   for d in ['left', 'right', 'up', 'down']}
        
        energies = []
        
        for iteration in range(self.max_iter):
            # Message passing step
            messages = self._update_messages(unary_cost, messages)
            messages = self._normalize_messages(messages)
            
            # Compute beliefs and decode
            beliefs = self._compute_beliefs(unary_cost, messages)
            labels = beliefs.argmin(dim=-1)
            
            # Evaluate energy
            energy = self._compute_energy(unary_cost, labels)
            energies.append(energy)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"  iter {iteration+1}/{self.max_iter}: energy = {energy:.2f}")
        
        return labels.cpu().numpy(), energies
    
    def _update_messages(self, unary, messages):
        """Update all messages for one iteration."""
        h, w, L = unary.shape
        new_messages = {}
        
        # Message from p to its right neighbor (exclude msg from right)
        incoming = unary[:, :-1] + messages['left'][:, :-1] + \
                   messages['up'][:, :-1] + messages['down'][:, :-1]
        new_messages['right'] = torch.zeros(h, w, L, device=self.device)
        new_messages['right'][:, 1:] = self._potts_min(incoming)
        
        # Message from p to its left neighbor (exclude msg from left)
        incoming = unary[:, 1:] + messages['right'][:, 1:] + \
                   messages['up'][:, 1:] + messages['down'][:, 1:]
        new_messages['left'] = torch.zeros(h, w, L, device=self.device)
        new_messages['left'][:, :-1] = self._potts_min(incoming)
        
        # Message from p to its bottom neighbor (exclude msg from down)
        incoming = unary[:-1] + messages['left'][:-1] + \
                   messages['right'][:-1] + messages['up'][:-1]
        new_messages['down'] = torch.zeros(h, w, L, device=self.device)
        new_messages['down'][1:] = self._potts_min(incoming)
        
        # Message from p to its top neighbor (exclude msg from up)
        incoming = unary[1:] + messages['left'][1:] + \
                   messages['right'][1:] + messages['down'][1:]
        new_messages['up'] = torch.zeros(h, w, L, device=self.device)
        new_messages['up'][:-1] = self._potts_min(incoming)
        
        return new_messages
    
    def _potts_min(self, cost):
        """
        Efficiently compute min-convolution with Potts model.
        
        For each label l: output[l] = min(cost[l], min_over_all(cost) + λ)
        This is O(L) instead of O(L²) for general pairwise costs.
        
        Args:
            cost: (..., L) tensor
        Returns:
            result: (..., L) tensor
        """
        min_cost = cost.min(dim=-1, keepdim=True)[0]
        return torch.minimum(cost, min_cost + self.lambda_smooth)
    
    def _normalize_messages(self, messages):
        """
        Subtract mean from messages for numerical stability.
        Doesn't change MAP solution (only relative values matter).
        """
        return {key: msg - msg.mean(dim=-1, keepdim=True) 
                for key, msg in messages.items()}
    
    def _compute_beliefs(self, unary, messages):
        """Belief = unary cost + sum of all incoming messages."""
        beliefs = unary.clone()
        for msg in messages.values():
            beliefs += msg
        return beliefs
    
    def _compute_energy(self, unary, labels):
        """Compute total energy of the labeling."""
        h, w = labels.shape
        
        # Unary term: sum of selected costs
        rows = torch.arange(h, device=labels.device)[:, None].expand(h, w)
        cols = torch.arange(w, device=labels.device)[None, :].expand(h, w)
        unary_energy = unary[rows, cols, labels].sum()
        
        # Pairwise term: count neighboring pairs with different labels
        horizontal_diff = (labels[:, :-1] != labels[:, 1:]).sum()
        vertical_diff = (labels[:-1] != labels[1:]).sum()
        pairwise_energy = self.lambda_smooth * (horizontal_diff + vertical_diff)
        
        return (unary_energy + pairwise_energy).item()
    
    def _to_tensor(self, x):
        """Convert input to torch tensor on device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected numpy array or torch tensor, got {type(x)}")
        return x.to(self.device)


# ============================================================================
# Stereo Matching Application
# ============================================================================

def compute_stereo_cost(left, right, dmax=16, tau=15):
    """
    Compute truncated L1 unary cost for stereo matching.
    
    Cost measures photometric dissimilarity between left[y,x] and right[y,x-d]
    for each disparity d. Lower cost = better match.
    
    Args:
        left: (H, W, 3) left image (RGB, 0-255 range)
        right: (H, W, 3) right image
        dmax: Maximum disparity to consider
        tau: Truncation threshold (robustness to outliers)
        
    Returns:
        (H, W, dmax) cost tensor
    """
    if isinstance(left, np.ndarray):
        left = torch.from_numpy(left).float()
        right = torch.from_numpy(right).float()
    
    h, w = left.shape[:2]
    device = left.device
    cost = torch.zeros(h, w, dmax, device=device)
    
    for d in range(dmax):
        if d == 0:
            # No shift
            diff = (left - right).abs().sum(dim=2) / 3.0
        else:
            # Shift right image left by d pixels
            diff = (left[:, d:] - right[:, :-d]).abs().sum(dim=2) / 3.0
            # Occluded region: assign high cost
            diff = torch.cat([torch.full((h, d), tau, device=device), diff], dim=1)
        
        cost[:, :, d] = diff.clamp(max=tau)
    
    return cost


# ============================================================================
# Image Segmentation Application  
# ============================================================================

def compute_segmentation_cost(image, num_classes, method='kmeans'):
    """
    Compute unary cost for image segmentation.
    
    Args:
        image: (H, W, 3) image array
        num_classes: Number of segments
        method: 'kmeans' (uses k-means on pixel colors)
        
    Returns:
        (H, W, num_classes) cost tensor
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    h, w, c = image.shape
    device = image.device
    
    # Flatten pixels for clustering
    pixels = image.reshape(-1, c)
    
    # Simple k-means initialization: evenly spaced centers in RGB space
    # NOTE: For real use, run actual k-means or use a classifier
    min_vals = pixels.min(dim=0)[0]
    max_vals = pixels.max(dim=0)[0]
    centers = torch.stack([
        min_vals + (max_vals - min_vals) * i / (num_classes - 1)
        for i in range(num_classes)
    ])
    
    # Compute distance to each center
    cost = torch.zeros(h, w, num_classes, device=device)
    for k in range(num_classes):
        dist = torch.norm(image - centers[k], dim=2)
        cost[:, :, k] = dist
    
    return cost


# ============================================================================
# Visualization
# ============================================================================

def plot_results(data, energies, title, save_path=None, cmap='jet'):
    """Plot result and energy curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Result visualization
    im = axes[0].imshow(data, cmap=cmap)
    axes[0].set_title(title)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])
    
    # Energy curve
    axes[1].plot(energies, linewidth=2, color='steelblue')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Convergence')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Examples
# ============================================================================

def example_stereo():
    """Example: Stereo matching on synthetic data."""
    print("="*60)
    print("STEREO MATCHING EXAMPLE")
    print("="*60)
    
    # Create synthetic stereo pair
    h, w = 80, 100
    left = np.random.rand(h, w, 3) * 50  # Low intensity background
    
    # Add objects at different depths
    left[20:40, 30:50] = [200, 100, 100]  # Red square
    left[45:65, 60:85] = [100, 200, 100]  # Green square
    
    # Create right image with disparities
    right = np.zeros_like(left)
    right[:, :w-5] = left[:, 5:]  # Background: disparity 5
    right[20:40, :w-10] = left[20:40, 10:]  # Red: disparity 10
    right[45:65, :w-7] = left[45:65, 7:]  # Green: disparity 7
    
    # Compute costs
    unary = compute_stereo_cost(left, right, dmax=16, tau=15)
    
    # Run LBP with different smoothness
    for lam in [0.5, 5.0, 50.0]:
        lbp = LoopyBeliefPropagation(lambda_smooth=lam, max_iter=40)
        disparity, energies = lbp.fit(unary)
        plot_results(disparity, energies, f'Disparity Map (λ={lam})',
                    save_path=f'stereo_lam{lam}.png')


def example_segmentation():
    """Example: Image segmentation."""
    print("\n" + "="*60)
    print("SEGMENTATION EXAMPLE")
    print("="*60)
    
    # Create image with three regions + noise
    img = np.zeros((80, 100, 3))
    img[:, :40] = [50, 50, 150]    # Blue region
    img[:, 40:70] = [150, 50, 50]  # Red region  
    img[:, 70:] = [50, 150, 50]    # Green region
    img += np.random.randn(80, 100, 3) * 20  # Add noise
    img = np.clip(img, 0, 255)
    
    # Compute costs
    unary = compute_segmentation_cost(img, num_classes=3)
    
    # Run LBP
    for lam in [0.5, 5.0]:
        lbp = LoopyBeliefPropagation(lambda_smooth=lam, max_iter=30)
        labels, energies = lbp.fit(unary)
        plot_results(labels, energies, f'Segmentation (λ={lam})',
                    save_path=f'segment_lam{lam}.png', cmap='tab10')


if __name__ == "__main__":
    example_stereo()
    example_segmentation()