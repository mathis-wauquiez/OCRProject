import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from typing import Dict, List
import warnings
import tqdm

from src.clustering.metrics import compute_metrics, UNKNOWN_LABEL


def extract_data(
    community_id,
    graph,
    dataframe
):
    node_list = list(graph.nodes())
    membership = dataframe['membership']

    node_indices = np.where(membership == community_id)[0]
    nodes_in_community = [node_list[i] for i in node_indices]
    subgraph_nx = graph.subgraph(nodes_in_community).copy()

    return dataframe.iloc[node_indices], subgraph_nx, node_indices


def evaluate_splitting_method(
    graph,
    dataframe,
    splitting_method,
    membership_col_name='membership_split',
    character_col_name='char_qwen_single',
    exclude_label=UNKNOWN_LABEL,  # Add exclude_label parameter
    use_tqdm=False
):
    membership = dataframe['membership']
    communities = np.unique(membership)
    if use_tqdm:
        communities = tqdm.tqdm(communities, desc="Splitting communities...", colour="magenta")

    total_communities = 0
    
    # Initialize the new column in the original dataframe
    dataframe[membership_col_name] = -1

    for community in communities:
        subdf, subgraph, node_indices = extract_data(community, graph, dataframe)

        new_membership = splitting_method(subdf, subgraph)  # MUST be 0 - (N-1)
                                                             # otherwise won't work as expected but won't crash

        unique_memberships = np.unique(new_membership)
        expected_range = np.arange(len(unique_memberships))
        assert np.array_equal(unique_memberships, expected_range), \
            f"Membership should be 0 to {len(unique_memberships)-1}, got {unique_memberships}"
        
        dataframe.loc[node_indices, membership_col_name] = new_membership + total_communities
        total_communities += len(unique_memberships)

    if use_tqdm:
        print('Computing the metrics ...')
    
    return compute_metrics(
        reference_labels=dataframe[character_col_name],
        predicted_labels=dataframe[membership_col_name],
        exclude_label=exclude_label  # Pass exclude_label to compute_metrics
    )


def evaluate_all_algorithms(
    graph,
    dataframe,
    algorithms: Dict,
    membership_col_base='membership',
    character_col_name='char_qwen_single',
    exclude_label=UNKNOWN_LABEL,  # Add exclude_label parameter
    use_tqdm=True,
    max_time_per_algo=300  # 5 minutes timeout per algorithm
):
    """
    Evaluate all clustering algorithms and collect their metrics.
    
    Returns:
        results_df: DataFrame with metrics for each algorithm
        timings: Dict with execution times
        failed_algos: List of algorithms that failed
    """
    results = []
    timings = {}
    failed_algos = []
    
    if use_tqdm:
        algo_iter = tqdm.tqdm(algorithms.items(), desc="Evaluating algorithms", colour="cyan")
    else:
        algo_iter = algorithms.items()
    
    for algo_name, algo_func in algo_iter:
        if use_tqdm:
            algo_iter.set_postfix_str(f"Running {algo_name}")
        
        try:
            # Measure execution time
            start_time = time()
            
            # Run evaluation
            metrics = evaluate_splitting_method(
                graph=graph,
                dataframe=dataframe,
                splitting_method=algo_func,
                membership_col_name=f'{membership_col_base}_{algo_name}',
                character_col_name=character_col_name,
                exclude_label=exclude_label,  # Pass exclude_label
                use_tqdm=False
            )
            
            elapsed_time = time() - start_time
            
            # Check for timeout
            if elapsed_time > max_time_per_algo:
                print(f"⚠️  {algo_name} exceeded time limit ({elapsed_time:.1f}s)")
            
            # Store results
            metrics['algorithm'] = algo_name
            metrics['time_seconds'] = elapsed_time
            results.append(metrics)
            timings[algo_name] = elapsed_time
            
        except Exception as e:
            print(f"❌ {algo_name} failed: {str(e)}")
            failed_algos.append(algo_name)
            continue
    
    if len(results) == 0:
        print("❌ No algorithms completed successfully!")
        return pd.DataFrame(), {}, failed_algos
    
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    cols_order = ['algorithm', 'time_seconds', 'n_samples', 'n_samples_total',
                  'n_clusters_predicted', 'adjusted_rand_index', 'normalized_mutual_info', 
                  'fowlkes_mallows_index', 'v_measure', 'homogeneity', 'completeness',
                  'purity', 'inverse_purity', 'f1_score', 'accuracy_optimal_match']
    
    # Keep only columns that exist
    cols_order = [col for col in cols_order if col in results_df.columns]
    other_cols = [col for col in results_df.columns if col not in cols_order]
    results_df = results_df[cols_order + other_cols]
    
    return results_df, timings, failed_algos


def create_comparison_plots(results_df, save_path=None):
    """
    Create comprehensive comparison plots for algorithm evaluation.
    """
    if len(results_df) == 0:
        print("No results to plot!")
        return None
    
    # Define key metrics to plot
    quality_metrics = [
        'adjusted_rand_index',
        'normalized_mutual_info',
        'fowlkes_mallows_index',
        'v_measure',
        'f1_score',
        'accuracy_optimal_match'
    ]
    
    # Filter to only available metrics
    quality_metrics = [m for m in quality_metrics if m in results_df.columns]
    
    # Create figure with subplots
    n_plots = len(quality_metrics) + 2  # metrics + time + clusters
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()
    
    # Sort by adjusted_rand_index for consistent ordering
    if 'adjusted_rand_index' in results_df.columns:
        sorted_df = results_df.sort_values('adjusted_rand_index', ascending=True)
    else:
        sorted_df = results_df.copy()
    
    # Plot each quality metric
    for idx, metric in enumerate(quality_metrics):
        ax = axes[idx]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_df))
        values = sorted_df[metric].values
        colors = plt.cm.viridis(values / (values.max() + 0.001))
        
        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['algorithm'].values, fontsize=8)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_xlim(0, max(1.0, values.max() * 1.1))
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.01:
                ax.text(val + 0.01, i, f'{val:.3f}', 
                       va='center', fontsize=7)
    
    # Plot execution time
    ax = axes[len(quality_metrics)]
    sorted_time = sorted_df.sort_values('time_seconds', ascending=True)
    y_pos = np.arange(len(sorted_time))
    times = sorted_time['time_seconds'].values
    
    colors = plt.cm.RdYlGn_r(times / (times.max() + 0.001))
    bars = ax.barh(y_pos, times, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_time['algorithm'].values, fontsize=8)
    ax.set_xlabel('Execution Time (seconds)', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, times)):
        ax.text(val + 0.1, i, f'{val:.1f}s', va='center', fontsize=7)
    
    # Plot number of clusters
    ax = axes[len(quality_metrics) + 1]
    sorted_clusters = sorted_df.sort_values('n_clusters_predicted', ascending=True)
    y_pos = np.arange(len(sorted_clusters))
    n_clusters = sorted_clusters['n_clusters_predicted'].values
    
    colors = plt.cm.plasma(n_clusters / (n_clusters.max() + 0.001))
    bars = ax.barh(y_pos, n_clusters, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_clusters['algorithm'].values, fontsize=8)
    ax.set_xlabel('Number of Clusters', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, n_clusters)):
        ax.text(val + 10, i, f'{val:.0f}', va='center', fontsize=7)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Algorithm Comparison: Clustering Performance Metrics', 
                 fontsize=16, fontweight='bold', y=1.002)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_heatmap(results_df, save_path=None):
    """
    Create a heatmap showing all metrics for all algorithms.
    """
    if len(results_df) == 0:
        print("No results to plot!")
        return None
    
    # Select metrics for heatmap
    metrics_to_plot = [
        'adjusted_rand_index', 'normalized_mutual_info', 
        'fowlkes_mallows_index', 'v_measure', 
        'homogeneity', 'completeness',
        'purity', 'inverse_purity', 
        'f1_score', 'accuracy_optimal_match'
    ]
    
    # Filter to available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in results_df.columns]
    
    if len(metrics_to_plot) == 0:
        print("No metrics available for heatmap!")
        return None
    
    # Create data matrix
    data = results_df.set_index('algorithm')[metrics_to_plot]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(results_df) * 0.4)))
    
    # Create heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Score'},
                ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Algorithms', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_pareto_plot(results_df, quality_metric='adjusted_rand_index', save_path=None):
    """
    Create a Pareto plot showing quality vs speed trade-off.
    """
    if len(results_df) == 0:
        print("No results to plot!")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(
        results_df['time_seconds'],
        results_df[quality_metric],
        s=results_df['n_clusters_predicted'] / 5,
        c=results_df[quality_metric],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5
    )
    
    # Add algorithm labels
    for _, row in results_df.iterrows():
        ax.annotate(
            row['algorithm'],
            (row['time_seconds'], row[quality_metric]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
        )
    
    ax.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel(quality_metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Speed Trade-off\n(bubble size = number of clusters)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(quality_metric.replace('_', ' ').title(), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def print_summary(results_df, top_n=5):
    """
    Print a formatted summary of the results.
    """
    if len(results_df) == 0:
        print("No results to summarize!")
        return
    
    print("=" * 100)
    print("CLUSTERING ALGORITHM EVALUATION SUMMARY".center(100))
    print("=" * 100)
    print()
    
    # Overall statistics
    print("📊 OVERALL STATISTICS")
    print("-" * 100)
    print(f"Total algorithms evaluated: {len(results_df)}")
    print(f"Total execution time: {results_df['time_seconds'].sum():.1f} seconds")
    print(f"Average clusters per algorithm: {results_df['n_clusters_predicted'].mean():.1f}")
    print(f"Cluster range: {results_df['n_clusters_predicted'].min():.0f} - {results_df['n_clusters_predicted'].max():.0f}")
    
    # Add information about excluded labels
    if 'n_samples' in results_df.columns and 'n_samples_total' in results_df.columns:
        avg_excluded = (results_df['n_samples_total'] - results_df['n_samples']).mean()
        print(f"Average excluded samples (unknown labels): {avg_excluded:.1f}")
    
    print()
    
    # Top performers
    if 'adjusted_rand_index' in results_df.columns:
        print(f"🏆 TOP {top_n} PERFORMERS (by Adjusted Rand Index)")
        print("-" * 100)
        top_algos = results_df.nlargest(top_n, 'adjusted_rand_index')
        
        for idx, row in enumerate(top_algos.iterrows(), 1):
            _, row = row
            print(f"\n{idx}. {row['algorithm']}")
            print(f"   ARI: {row['adjusted_rand_index']:.4f} | "
                  f"NMI: {row['normalized_mutual_info']:.4f} | "
                  f"F1: {row['f1_score']:.4f}")
            print(f"   Clusters: {row['n_clusters_predicted']:.0f} | "
                  f"Time: {row['time_seconds']:.2f}s")
    
    print()
    
    # Fastest algorithms
    print(f"⚡ FASTEST {top_n} ALGORITHMS")
    print("-" * 100)
    fastest = results_df.nsmallest(top_n, 'time_seconds')
    
    for idx, row in enumerate(fastest.iterrows(), 1):
        _, row = row
        quality = row.get('adjusted_rand_index', row.get('f1_score', 0))
        print(f"{idx}. {row['algorithm']:40s} - {row['time_seconds']:6.2f}s (quality: {quality:.4f})")
    
    print()
    
    # Balanced performance (quality/time ratio)
    if 'adjusted_rand_index' in results_df.columns:
        results_df['efficiency'] = results_df['adjusted_rand_index'] / (results_df['time_seconds'] + 1)
        
        print(f"⚖️  BEST {top_n} BALANCED (Quality/Time Efficiency)")
        print("-" * 100)
        balanced = results_df.nlargest(top_n, 'efficiency')
        
        for idx, row in enumerate(balanced.iterrows(), 1):
            _, row = row
            print(f"{idx}. {row['algorithm']:40s} - "
                  f"ARI: {row['adjusted_rand_index']:.4f} in {row['time_seconds']:.2f}s "
                  f"(eff: {row['efficiency']:.4f})")
    
    print()
    print("=" * 100)


def generate_report(results_df, output_dir='.'):
    """
    Generate a complete evaluation report with plots and summary.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary
    print_summary(results_df)
    
    if len(results_df) == 0:
        print("\n❌ No results to generate report from!")
        return None, None, None
    
    # Create plots
    print("\n📈 Generating comparison plots...")
    fig1 = create_comparison_plots(results_df, save_path=f'{output_dir}/comparison_plots.png')
    
    print("🔥 Generating heatmap...")
    fig2 = create_heatmap(results_df, save_path=f'{output_dir}/heatmap.png')
    
    print("📊 Generating Pareto plot...")
    fig3 = create_pareto_plot(results_df, save_path=f'{output_dir}/pareto_plot.png')
    
    # Save detailed results to CSV
    results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    print(f"\n💾 Detailed results saved to {output_dir}/detailed_results.csv")
    
    plt.show()
    
    return fig1, fig2, fig3


# Example usage:
if __name__ == "__main__":
    # Assuming these are defined in your environment:
    # - graph: NetworkX graph
    # - patches_df: DataFrame with clustering features
    # - algorithms: Dict of algorithm functions
    
    # Run evaluation
    results_df, timings, failed_algos = evaluate_all_algorithms(
        graph=graph,
        dataframe=patches_df,
        algorithms=algorithms,
        character_col_name='char_qwen_single',
        exclude_label=UNKNOWN_LABEL,
        use_tqdm=True
    )
    
    # Report failed algorithms
    if failed_algos:
        print(f"\n⚠️  {len(failed_algos)} algorithms failed:")
        for algo in failed_algos:
            print(f"  - {algo}")
    
    # Generate complete report
    figs = generate_report(results_df, output_dir='./outputs/evaluation/clustering')