# python analyze_clustering_sweep_results.py
import json
import pandas as pd
from pathlib import Path
import sys
import yaml


def is_sweep_directory(path: Path) -> bool:
    """Check if a directory is a valid Hydra sweep directory"""
    if not path.is_dir():
        return False
    
    # Check for multirun.yaml (Hydra sweep marker)
    if (path / "multirun.yaml").exists():
        return True
    
    # Alternative: check if it has numbered subdirectories with expected structure
    numbered_dirs = [d for d in path.iterdir() 
                     if d.is_dir() and d.name.isdigit()]
    
    if numbered_dirs:
        # Check if at least one has the expected structure
        sample_dir = numbered_dirs[0]
        has_structure = (
            (sample_dir / "metrics.json").exists() or
            (sample_dir / "config.yaml").exists() or
            (sample_dir / ".hydra").exists()
        )
        return has_structure
    
    return False


def find_sweep_directories(base_path: Path) -> list[Path]:
    """Find all Hydra sweep directories recursively"""
    sweep_dirs = []
    
    # Search pattern: outputs/YYYY-MM-DD/HH-MM-SS/
    for date_dir in base_path.glob("*"):
        if not date_dir.is_dir():
            continue
        
        for time_dir in date_dir.glob("*"):
            if is_sweep_directory(time_dir):
                sweep_dirs.append(time_dir)
    
    return sorted(sweep_dirs, key=lambda x: x.stat().st_mtime)


def collect_sweep_results(sweep_dir: str):
    """Collect all metrics from a sweep run"""
    sweep_path = Path(sweep_dir)
    
    if not sweep_path.exists():
        print(f"Error: Directory {sweep_path} does not exist!")
        return None
    
    if not is_sweep_directory(sweep_path):
        print(f"Warning: {sweep_path} doesn't look like a Hydra sweep directory")
        print("Attempting to process anyway...")
    
    results = []
    errors = []
    
    # Find all numbered run directories
    run_dirs = [d for d in sweep_path.iterdir() 
                if d.is_dir() and d.name.isdigit()]
    
    if not run_dirs:
        print(f"Error: No numbered run directories found in {sweep_path}")
        return None
    
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in sorted(run_dirs, key=lambda x: int(x.name)):
        metrics_file = run_dir / "metrics.json"
        config_file = run_dir / "config.yaml"
        
        if not metrics_file.exists():
            errors.append(f"{run_dir.name}: missing metrics.json")
            continue
        
        if not config_file.exists():
            errors.append(f"{run_dir.name}: missing config.yaml")
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract parameters from config
            result = {
                'run_id': int(run_dir.name),
                'epsilon': config['method']['featureMatcher']['params']['epsilon'],
                'gamma': config['method']['community_detection']['gamma'],
            }
            
            # Add all metrics (flatten if needed)
            if 'error' in metrics:
                result['error'] = metrics['error']
                errors.append(f"{run_dir.name}: {metrics['error']}")
            else:
                result.update(metrics)
            
            results.append(result)
            
        except Exception as e:
            errors.append(f"{run_dir.name}: {str(e)}")
            continue
    
    if errors:
        print("\nWarnings/Errors encountered:")
        for err in errors:
            print(f"  - {err}")
    
    if not results:
        print("No valid results found!")
        return None
    
    df = pd.DataFrame(results)
    
    # Sort by parameters for easier viewing
    df = df.sort_values(['epsilon', 'gamma']).reset_index(drop=True)
    
    # Save aggregated results
    output_csv = sweep_path / "all_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved aggregated results to {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Successful runs: {len(df[~df.get('error', pd.Series()).notna()])}")
    print(f"Parameters swept:")
    print(f"  - epsilon: {sorted(df['epsilon'].unique())}")
    print(f"  - gamma: {sorted(df['gamma'].unique())}")
    
    # Show top results by key metrics
    metric_columns = [col for col in df.columns 
                     if col not in ['run_id', 'epsilon', 'gamma', 'error']]
    
    if 'adjusted_rand_index' in metric_columns:
        print("\n" + "-"*80)
        print("TOP 5 RESULTS BY ADJUSTED RAND INDEX:")
        print("-"*80)
        display_cols = ['run_id', 'epsilon', 'gamma', 'adjusted_rand_index']
        if 'normalized_mutual_info' in metric_columns:
            display_cols.append('normalized_mutual_info')
        if 'v_measure' in metric_columns:
            display_cols.append('v_measure')
        
        top_ari = df.nlargest(5, 'adjusted_rand_index')[display_cols]
        print(top_ari.to_string(index=False))
    
    if 'v_measure' in metric_columns:
        print("\n" + "-"*80)
        print("TOP 5 RESULTS BY V-MEASURE:")
        print("-"*80)
        display_cols = ['run_id', 'epsilon', 'gamma', 'v_measure']
        if 'homogeneity' in metric_columns:
            display_cols.append('homogeneity')
        if 'completeness' in metric_columns:
            display_cols.append('completeness')
        
        top_vm = df.nlargest(5, 'v_measure')[display_cols]
        print(top_vm.to_string(index=False))
    
    # Save best configurations
    if 'adjusted_rand_index' in metric_columns:
        valid_df = df[df['adjusted_rand_index'].notna()]
        if len(valid_df) > 0:
            best_row = valid_df.loc[valid_df['adjusted_rand_index'].idxmax()]
            best_config = {
                'best_by_ari': {
                    'epsilon': float(best_row['epsilon']),
                    'gamma': float(best_row['gamma']),
                    'metrics': {k: float(v) for k, v in best_row.items() 
                               if k not in ['run_id', 'epsilon', 'gamma', 'error'] 
                               and pd.notna(v)}
                }
            }
            
            best_config_file = sweep_path / "best_config.yaml"
            with open(best_config_file, 'w') as f:
                yaml.dump(best_config, f, default_flow_style=False)
            print(f"\nBest configuration saved to {best_config_file}")
    
    return df


def plot_sweep_results(df: pd.DataFrame, sweep_path: Path):
    """
    Create visualization plots for sweep results.
    Requires matplotlib and seaborn.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return
    
    # Set style
    sns.set_style("whitegrid")
    
    # Metrics to plot
    metric_cols = [col for col in df.columns 
                   if col not in ['run_id', 'epsilon', 'gamma', 'error'] 
                   and df[col].notna().any()]
    
    if not metric_cols:
        print("No metrics to plot")
        return
    
    print(f"\nGenerating {len(metric_cols)} heatmap plots...")
    
    # Create heatmaps for each metric
    for metric in metric_cols:
        # Pivot table for heatmap
        pivot = df.pivot_table(
            values=metric,
            index='gamma',
            columns='epsilon',
            aggfunc='mean'
        )
        
        if pivot.empty or pivot.isna().all().all():
            continue
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=pivot.values[~pd.isna(pivot.values)].mean())
        plt.title(f'{metric.replace("_", " ").title()} by Epsilon and Gamma')
        plt.xlabel('Epsilon')
        plt.ylabel('Gamma')
        plt.tight_layout()
        
        plot_file = sweep_path / f"heatmap_{metric}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {sweep_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sweep_dir = sys.argv[1]
        df = collect_sweep_results(sweep_dir)
    else:
        # Find all sweep directories
        outputs = Path("outputs/clustering")
        if not outputs.exists():
            print("No outputs directory found!")
            sys.exit(1)
        
        sweep_dirs = find_sweep_directories(outputs)
        
        if not sweep_dirs:
            print("No sweep directories found in outputs/!")
            print("\nSearched for directories matching pattern: outputs/YYYY-MM-DD/HH-MM-SS/")
            sys.exit(1)
        
        print(f"Found {len(sweep_dirs)} sweep directories:")
        for i, sd in enumerate(sweep_dirs, 1):
            print(f"  {i}. {sd.relative_to(outputs)}")
        
        # Use most recent
        sweep_dir = sweep_dirs[-1]
        print(f"\nUsing most recent: {sweep_dir.relative_to(outputs)}")
        
        df = collect_sweep_results(sweep_dir)
    
    if df is not None:
        # Optional: create plots
        plot_sweep_results(df, Path(sweep_dir))
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)