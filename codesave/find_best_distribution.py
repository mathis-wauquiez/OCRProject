"""
Find the best distribution family across multiple samples.

For data of shape (N_distrs, N_observations) where N_distrs is large (e.g., 3000),
this script fits each candidate distribution to each sample and aggregates the results
to determine which distribution family performs best overall.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def fit_and_evaluate_distribution(sample, dist_class):
    """
    Fit a distribution to a sample and return goodness-of-fit metrics.
    
    Returns:
    --------
    dict with keys: ks_statistic, ks_p_value, aic, bic, log_likelihood, fit_success
    """
    try:
        # Fit the distribution
        params = dist_class.fit(sample)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = kstest(sample, dist_class(*params).cdf)
        
        # Calculate log-likelihood
        log_likelihood = np.sum(dist_class.logpdf(sample, *params))
        
        # Handle invalid log-likelihood
        if not np.isfinite(log_likelihood):
            return {
                'ks_statistic': np.nan,
                'ks_p_value': 0.0,
                'aic': np.inf,
                'bic': np.inf,
                'log_likelihood': -np.inf,
                'fit_success': False
            }
        
        # Calculate AIC and BIC
        k = len(params)  # number of parameters
        n = len(sample)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        return {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'fit_success': True
        }
        
    except Exception as e:
        return {
            'ks_statistic': np.nan,
            'ks_p_value': 0.0,
            'aic': np.inf,
            'bic': np.inf,
            'log_likelihood': -np.inf,
            'fit_success': False
        }


def find_best_distribution_family(data, candidate_distributions=None, 
                                   n_samples=None, verbose=True):
    """
    Find the best distribution family across all samples.
    
    Parameters:
    -----------
    data : np.ndarray, shape (N_distrs, N_observations)
        Your collection of samples
    candidate_distributions : list of tuples
        [(name, scipy.stats.distribution), ...] to test
    n_samples : int, optional
        If provided, only use first n_samples (for testing)
    verbose : bool
        Show progress bar
        
    Returns:
    --------
    results_df : DataFrame with aggregated results
    detailed_df : DataFrame with per-sample results (optional)
    """
    
    if candidate_distributions is None:
        candidate_distributions = [
            ('normal', stats.norm),
            ('exponential', stats.expon),
            ('gamma', stats.gamma),
            ('lognormal', stats.lognorm),
            ('weibull', stats.weibull_min),
            ('beta', stats.beta),
        ]
    
    N_distrs = data.shape[0] if n_samples is None else min(n_samples, data.shape[0])
    
    print(f"Analyzing {N_distrs} distributions...")
    print(f"Testing {len(candidate_distributions)} candidate distribution families\n")
    
    # Data validation and diagnostics
    print("="*80)
    print("DATA DIAGNOSTICS")
    print("="*80)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"\nSample statistics (first sample):")
    sample_0 = data[0]
    print(f"  Min: {np.min(sample_0):.4f}")
    print(f"  Max: {np.max(sample_0):.4f}")
    print(f"  Mean: {np.mean(sample_0):.4f}")
    print(f"  Std: {np.std(sample_0):.4f}")
    print(f"\nData quality checks:")
    n_nan = np.sum(np.isnan(data))
    n_inf = np.sum(np.isinf(data))
    n_negative = np.sum(data < 0)
    n_zero = np.sum(data == 0)
    print(f"  NaN values: {n_nan}")
    print(f"  Inf values: {n_inf}")
    print(f"  Negative values: {n_negative}")
    print(f"  Zero values: {n_zero}")
    
    if n_nan > 0 or n_inf > 0:
        print("\n⚠️  WARNING: Data contains NaN or Inf values!")
        print("   Consider cleaning your data first.")
    
    if n_negative > 0:
        print("\n⚠️  WARNING: Data contains negative values!")
        print("   Some distributions (exponential, lognormal, gamma, weibull) require positive data.")
        print("   Consider using only 'normal' distribution or transforming your data.")
    
    if n_zero > 0:
        print("\n⚠️  WARNING: Data contains exact zeros!")
        print("   This may cause issues with log-based distributions (lognormal).")
    
    print("="*80 + "\n")
    
    # Store results for each distribution and each sample
    all_results = []
    
    # Iterate through all samples
    iterator = tqdm(range(N_distrs), desc="Processing samples") if verbose else range(N_distrs)
    
    for i in iterator:
        sample = data[i]
        
        for dist_name, dist_class in candidate_distributions:
            result = fit_and_evaluate_distribution(sample, dist_class)
            result['sample_id'] = i
            result['distribution'] = dist_name
            all_results.append(result)
    
    # Create detailed results DataFrame
    detailed_df = pd.DataFrame(all_results)
    
    # Print fitting success summary
    print("\n" + "="*80)
    print("FITTING SUCCESS SUMMARY")
    print("="*80)
    for dist_name, _ in candidate_distributions:
        dist_results = detailed_df[detailed_df['distribution'] == dist_name]
        n_success = dist_results['fit_success'].sum()
        n_total = len(dist_results)
        success_rate = n_success / n_total * 100 if n_total > 0 else 0
        print(f"{dist_name:15s}: {n_success:4d}/{n_total:4d} successful ({success_rate:5.1f}%)")
    print("="*80 + "\n")
    
    # Aggregate results by distribution family
    print("\n" + "="*80)
    print("AGGREGATED RESULTS BY DISTRIBUTION FAMILY")
    print("="*80 + "\n")
    
    aggregated_results = []
    
    for dist_name, dist_class in candidate_distributions:
        dist_data = detailed_df[detailed_df['distribution'] == dist_name]
        
        # Filter out failed fits
        successful = dist_data[dist_data['fit_success']]
        n_success = len(successful)
        n_total = len(dist_data)
        success_rate = n_success / n_total if n_total > 0 else 0
        
        if n_success > 0:
            # Calculate aggregate metrics
            mean_aic = successful['aic'].mean()
            mean_bic = successful['bic'].mean()
            median_aic = successful['aic'].median()
            median_bic = successful['bic'].median()
            
            # Proportion of samples where this distribution passes KS test (p > 0.05)
            ks_pass_rate = (successful['ks_p_value'] > 0.05).sum() / n_success
            
            # Mean KS statistic (lower is better)
            mean_ks_stat = successful['ks_statistic'].mean()
            
            # Count how many times this distribution had the best AIC for each sample
            best_aic_count = 0
            for sample_id in successful['sample_id'].unique():
                sample_results = detailed_df[
                    (detailed_df['sample_id'] == sample_id) & 
                    (detailed_df['fit_success'] == True)
                ]
                if len(sample_results) > 0:
                    best_dist = sample_results.loc[sample_results['aic'].idxmin(), 'distribution']
                    if best_dist == dist_name:
                        best_aic_count += 1
            
            best_aic_rate = best_aic_count / n_success if n_success > 0 else 0
            
            aggregated_results.append({
                'distribution': dist_name,
                'success_rate': success_rate,
                'n_successful_fits': n_success,
                'mean_aic': mean_aic,
                'median_aic': median_aic,
                'mean_bic': mean_bic,
                'median_bic': median_bic,
                'mean_ks_statistic': mean_ks_stat,
                'ks_pass_rate': ks_pass_rate,
                'best_aic_rate': best_aic_rate,
                'best_aic_count': best_aic_count
            })
    
    results_df = pd.DataFrame(aggregated_results)
    
    # Check if we have any results
    if len(results_df) == 0:
        print("\n" + "="*80)
        print("ERROR: No distributions were successfully fitted to any samples!")
        print("="*80)
        print("\nPossible issues:")
        print("  1. Data contains invalid values (NaN, inf)")
        print("  2. Data contains negative values (incompatible with some distributions)")
        print("  3. Data contains zeros (problematic for log-based distributions)")
        print("  4. Sample size is too small")
        print("\nPlease check your data and try again.")
        return pd.DataFrame(), detailed_df
    
    # Sort by mean AIC (lower is better)
    results_df = results_df.sort_values('mean_aic').reset_index(drop=True)
    
    # Display results
    print("Metrics explanation:")
    print("  - mean_aic/median_aic: Lower is better (average fit quality)")
    print("  - ks_pass_rate: Proportion passing KS test (higher is better)")
    print("  - best_aic_rate: Proportion of samples where this dist has lowest AIC")
    print("  - success_rate: Proportion of samples successfully fitted\n")
    
    print(results_df.to_string(index=False))
    
    # Identify the winner
    best_dist = results_df.iloc[0]['distribution']
    print("\n" + "="*80)
    print(f"🏆 BEST DISTRIBUTION FAMILY: {best_dist.upper()}")
    print("="*80)
    print(f"  - Mean AIC: {results_df.iloc[0]['mean_aic']:.2f}")
    print(f"  - Best AIC in {results_df.iloc[0]['best_aic_rate']*100:.1f}% of samples")
    print(f"  - KS test pass rate: {results_df.iloc[0]['ks_pass_rate']*100:.1f}%")
    print("="*80 + "\n")
    
    return results_df, detailed_df


def plot_comparison(results_df, detailed_df, output_path='/mnt/user-data/outputs/'):
    """
    Create visualization comparing distribution families.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean AIC comparison
    ax = axes[0, 0]
    results_sorted = results_df.sort_values('mean_aic')
    ax.barh(results_sorted['distribution'], results_sorted['mean_aic'])
    ax.set_xlabel('Mean AIC (lower is better)')
    ax.set_title('Mean AIC by Distribution Family')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Best AIC rate
    ax = axes[0, 1]
    results_sorted = results_df.sort_values('best_aic_rate', ascending=False)
    ax.barh(results_sorted['distribution'], results_sorted['best_aic_rate'] * 100)
    ax.set_xlabel('% of samples where distribution has best AIC')
    ax.set_title('Best AIC Rate by Distribution Family')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. KS test pass rate
    ax = axes[1, 0]
    results_sorted = results_df.sort_values('ks_pass_rate', ascending=False)
    ax.barh(results_sorted['distribution'], results_sorted['ks_pass_rate'] * 100)
    ax.set_xlabel('% of samples passing KS test (p > 0.05)')
    ax.set_title('KS Test Pass Rate by Distribution Family')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. AIC distribution (violin plot)
    ax = axes[1, 1]
    
    # Prepare data for violin plot (limit to reasonable AIC values)
    plot_data = []
    plot_labels = []
    for dist in results_df['distribution']:
        dist_aic = detailed_df[
            (detailed_df['distribution'] == dist) & 
            (detailed_df['fit_success'] == True) &
            (detailed_df['aic'] < detailed_df['aic'].quantile(0.95))  # Remove outliers
        ]['aic'].values
        if len(dist_aic) > 0:  # Only include if we have data
            plot_data.append(dist_aic)
            plot_labels.append(dist)
    
    if len(plot_data) > 0:
        parts = ax.violinplot(plot_data, vert=False, showmeans=True, showmedians=True)
        ax.set_yticks(range(1, len(plot_labels) + 1))
        ax.set_yticklabels(plot_labels)
        ax.set_xlabel('AIC (lower is better)')
        ax.set_title('AIC Distribution (95th percentile cutoff)')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}distribution_family_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}distribution_family_comparison.png")
    
    return fig


# Example usage
if __name__ == "__main__":
    
    # Generate example data with 3000 distributions
    # In reality, each might be from the same family with different parameters
    np.random.seed(42)
    N_distrs = 3000
    N_observations = 100
    
    print("Generating example data...")
    data = np.zeros((N_distrs, N_observations))
    
    # Simulate: most are actually gamma distributions with varying parameters
    for i in range(N_distrs):
        shape = np.random.uniform(1, 5)
        scale = np.random.uniform(0.5, 3)
        data[i] = np.random.gamma(shape, scale, N_observations)
    
    print(f"Data shape: {data.shape}\n")
    
    # Test on a subset first (faster for demonstration)
    print("Running analysis on first 200 samples (set n_samples=None for all)...\n")
    results_df, detailed_df = find_best_distribution_family(
        data, 
        n_samples=200,  # Use None to analyze all 3000
        verbose=True
    )
    
    # Save results
    results_df.to_csv('/mnt/user-data/outputs/distribution_family_results.csv', index=False)
    detailed_df.to_csv('/mnt/user-data/outputs/detailed_fit_results.csv', index=False)
    print("\nResults saved to CSV files")
    
    # Create visualizations
    plot_comparison(results_df, detailed_df)