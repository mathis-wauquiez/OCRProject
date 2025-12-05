"""
QUICK START SCRIPT - Run this with your data!

This script will:
1. Check your data for issues
2. Automatically choose compatible distributions  
3. Run the fitting analysis
4. Show you the results
"""

import numpy as np
from scipy import stats
import sys


def auto_fit_distributions(data, test_first=True):
    """
    Automatically diagnose data and fit distributions.
    
    Parameters:
    -----------
    data : np.ndarray, shape (N_distrs, N_observations)
        Your distribution data
    test_first : bool
        If True, tests on first 100 samples before running full analysis
    """
    
    print("\n" + "="*80)
    print("AUTOMATIC DISTRIBUTION FITTING")
    print("="*80)
    
    # Quick checks
    print(f"\n1. DATA OVERVIEW")
    print(f"   Shape: {data.shape}")
    print(f"   N distributions: {data.shape[0]}")
    print(f"   N observations per distribution: {data.shape[1]}")
    
    # Check data quality
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    has_negative = np.any(data < 0)
    has_zero = np.any(data == 0)
    all_positive = np.all(data > 0)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    print(f"\n2. DATA QUALITY")
    print(f"   Range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Has negative: {has_negative}")
    print(f"   Has zeros: {has_zero}")
    
    # Handle critical issues
    if has_nan or has_inf:
        print("\n❌ ERROR: Data contains NaN or Inf values!")
        print("   Please clean your data first:")
        print("   data_clean = data[~np.isnan(data) & ~np.isinf(data)]")
        return None, None
    
    # Automatically choose distributions
    print(f"\n3. SELECTING COMPATIBLE DISTRIBUTIONS")
    
    if has_negative:
        print("   → Data has negative values")
        print("   → Using: Normal distribution only")
        candidate_distributions = [('normal', stats.norm)]
        data_to_fit = data
    elif has_zero:
        print("   → Data has zeros but no negatives")
        print("   → Adding small offset to avoid log(0)")
        print("   → Using: Normal, Gamma, Weibull")
        data_to_fit = data + 1e-10
        candidate_distributions = [
            ('normal', stats.norm),
            ('gamma', stats.gamma),
            ('weibull', stats.weibull_min),
        ]
    elif all_positive:
        print("   → Data is all positive")
        print("   → Using: Normal, Exponential, Gamma, Lognormal, Weibull")
        data_to_fit = data
        candidate_distributions = [
            ('normal', stats.norm),
            ('exponential', stats.expon),
            ('gamma', stats.gamma),
            ('lognormal', stats.lognorm),
            ('weibull', stats.weibull_min),
        ]
    else:
        print("   → Using: Normal, Gamma")
        data_to_fit = data
        candidate_distributions = [
            ('normal', stats.norm),
            ('gamma', stats.gamma),
        ]
    
    # Import the fitting function
    try:
        from find_best_distribution import find_best_distribution_family
    except ImportError:
        print("\n❌ ERROR: Cannot import find_best_distribution module")
        print("   Make sure find_best_distribution.py is in the same directory")
        return None, None
    
    # Test on subset first if requested
    if test_first and data.shape[0] > 100:
        print(f"\n4. TESTING ON FIRST 100 SAMPLES")
        print("   (to verify everything works before full run)")
        
        results_test, _ = find_best_distribution_family(
            data_to_fit,
            candidate_distributions=candidate_distributions,
            n_samples=100,
            verbose=False
        )
        
        if len(results_test) == 0:
            print("\n❌ Test failed - check your data more carefully")
            print("   Run: from data_diagnostics import diagnose_data")
            print("        diagnose_data(data)")
            return None, None
        
        print(f"\n   ✓ Test successful!")
        print(f"   Best distribution in test: {results_test.iloc[0]['distribution']}")
        
        # Ask to continue
        print(f"\n   Continue with all {data.shape[0]} samples? (This may take several minutes)")
        response = input("   Enter 'y' to continue or 'n' to stop [y/n]: ").strip().lower()
        if response != 'y':
            print("\n   Stopped by user. Test results:")
            return results_test, None
    
    # Run full analysis
    print(f"\n5. RUNNING FULL ANALYSIS ON {data.shape[0]} SAMPLES")
    print("   This may take a few minutes...")
    
    results_df, detailed_df = find_best_distribution_family(
        data_to_fit,
        candidate_distributions=candidate_distributions,
        n_samples=None,
        verbose=True
    )
    
    if len(results_df) == 0:
        print("\n❌ Analysis failed - all fits unsuccessful")
        return None, None
    
    # Show results
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    
    best = results_df.iloc[0]
    print(f"\n🏆 BEST DISTRIBUTION: {best['distribution'].upper()}")
    print(f"   Mean AIC: {best['mean_aic']:.2f}")
    print(f"   Best in {best['best_aic_rate']*100:.1f}% of samples")
    print(f"   KS test pass rate: {best['ks_pass_rate']*100:.1f}%")
    
    print(f"\nFull results:")
    print(results_df.to_string(index=False))
    
    return results_df, detailed_df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print("""
import numpy as np
from quick_start import auto_fit_distributions

# Load your data
data = np.load('your_data.npy')  # shape: (N_distrs, N_observations)

# Run automatic fitting
results_df, detailed_df = auto_fit_distributions(data, test_first=True)

# View the winner
if results_df is not None:
    print(f"Best distribution: {results_df.iloc[0]['distribution']}")
    
    # Save results
    results_df.to_csv('distribution_results.csv', index=False)
    if detailed_df is not None:
        detailed_df.to_csv('detailed_results.csv', index=False)
    """)
    print("="*80 + "\n")