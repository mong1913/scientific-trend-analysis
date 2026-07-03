import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.metrics import silhouette_samples, silhouette_score

def category_plot(cat_counts_pd, category_list, df_pivot_top):
    """
    Create topic bar chart and stacked area plot over time

    Parameters:
        cat_counts_pd : pandas DataFrame containing category counts
        category_list : string, column name for category list
        df_pivot_top : pandas DataFrame containing category counts pivoted by year
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7))
    plt.subplots_adjust(wspace=0.5)

    # Plot 1
    sns.barplot(data=cat_counts_pd, x=category_list, y='count', ax=ax1)
    ax1.set_xlabel('Category list', fontsize=18)
    ax1.set_ylabel('Paper Count', fontsize=18)
    ax1.tick_params(axis='x', labelsize=18, rotation=45)
    ax1.tick_params(axis='y', labelsize=18)

    # Plot 2
    df_pivot_top.plot(kind='area', stacked=True, ax=ax2, alpha=0.8)
    ax2.set_xlabel('Year', fontsize=18)
    ax2.set_ylabel('Number of Papers', fontsize=18)
    ax2.set_title('Papers Published Over Years by Category (Top 10 Categories)', fontsize=15, fontweight='bold')
    ax2.legend(title='Category', loc='upper left', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.tick_params(axis='x', labelsize=18, rotation=45)
    ax2.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.show()



def plot_dim_reduction(embedding_2d, cluster_labels, technique_name, 
                       cluster_id, sample_size, clustering_algo, figsize=(10, 7)):
    """
    Create a scatter plot to visualize clustering results in 2D using dimensionality reduction techniques
    
    Parameters:
        embedding_2d : ndarray, shape (n_samples, 2)
            2D coordinates from dimensionality reduction technique
        cluster_labels : array-like
            Cluster assignments for each sample
        technique_name : str
            Name of the technique (e.g., 'MDS', 'LLE', 't-SNE')
        cluster_id : str
            Name of the cluster id column
        sample_size : int
            Number of samples being visualized
        figsize : tuple, optional
            Figure size (width, height)
        clustering_algo: string, algorithm name (e.g. 'KMeans', 'GaussianMixture')
    """
    # Create DataFrame with consistent column naming
    col_names = [f'{technique_name}1', f'{technique_name}2']
    df_plot = pd.DataFrame(embedding_2d, columns=col_names)
    df_plot[cluster_id] = cluster_labels
    
    n_clusters = len(df_plot[cluster_id].unique())
    
    # Define distinct colors using tab20 colormap
    colors = plt.get_cmap('tab20').colors[:n_clusters]
    
    # Define marker shapes (cycle through if more clusters than shapes)
    # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster with unique color and marker
    for i, cluster in enumerate(sorted(df_plot[cluster_id].unique())):
        cluster_data = df_plot[df_plot[cluster_id] == cluster]
        ax.scatter(
            cluster_data[col_names[0]], 
            cluster_data[col_names[1]],
            c=[colors[i % len(colors)]], 
            #marker=markers[i % len(markers)],
            s=25, 
            alpha=0.7,
            label=f'{cluster}',
            edgecolors='none'
        )
    
    # plt.title(f'{technique_name} Visualization with {clustering_algo} Clusters (stratified n={sample_size:,})', 
    #           fontsize=12, fontweight='bold')
    plt.xlabel(f'{technique_name} Dimension 1', fontsize=11)
    plt.ylabel(f'{technique_name} Dimension 2', fontsize=11)
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left', 
               markerscale=1.5, frameon=True, fancybox=True)
    plt.tight_layout()
    plt.show()
    
    print(f"{technique_name} visualization complete with {n_clusters} clusters")
    
    return df_plot


def silhouette_plot(lsa_result, cluster_labels, best_k):
    """
    Create a silhouette plot to visualize clustering results
    
    Parameters:
        lsa_result : ndarray, shape (n_samples, n_components), ransformed data using LSA
        cluster_labels : array, cluster assignments for each sample
        best_k : int, best number of clusters determined by silhouette score
    """

    #Calculate the silhouette score for each data
    each_silhouette_score = silhouette_samples(lsa_result, cluster_labels, metric="euclidean")
    silhouette_avg = silhouette_score(lsa_result, cluster_labels)

    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    y_lower =10
    for i in range(best_k):
        ith_cluster_silhouette_values = each_silhouette_score[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        colors = plt.get_cmap('tab20').colors[:best_k]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7
        )
        
        # Label cluster in the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10, fontweight='bold')
        
        # Update y_lower for next cluster
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot (k={best_k}, sampled n={len(lsa_result):,})", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Silhouette Coefficient", fontsize=14)
    ax.set_ylabel("Cluster Label", fontsize=14)

    # Average silhouette line
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
            label=f'Average Score: {silhouette_avg:.4f}')

    ax.set_yticks([])
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def aic_bic_silhouette_for_gmm(k_range, aic_scores, bic_scores, silhouette_scores):
    """
    Plot AIC, BIC, and silhouette scores for GMM clustering
    
    Parameters:
        k_range : array, range of k values for GMM clustering
        aic_scores : array, AIC scores for each k value
        bic_scores : array, BIC scores for each k value
        silhouette_scores : array, silhouette scores for each k value
    """
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot AIC and BIC scores on the first y-axis
    ax1.plot(k_range, aic_scores, marker='o', label='AIC')
    ax1.plot(k_range, bic_scores, marker='o', label='BIC')

    # Add silhouette score to the plot on a second y-axis
    silhouette_score_values = [silhouette_scores[k] for k in k_range]
    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouette_score_values, marker='o', color='green', label='Silhouette Score')

    # best_k_bic = k_range[np.argmin(bic_scores)]
    # ax1.axvline(x=best_k_bic, color='red', linestyle='--', label=f'Best K (BIC): {best_k_bic}')

    ax1.set_xlabel('Number of cluster (k)', fontsize=14)
    ax1.set_ylabel('AIC / BIC', fontsize=14)
    ax2.set_ylabel('Silhouette Score', fontsize=14)
    ax1.set_title('GMM Model Selection Metrics', fontsize=16)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.show()

def probability_heatmap(probabilities):
    """
    Create a heatmap to visualize the probability of each document belonging to each cluster
    
    Parameters:
        probabilities_diag : ndarray,
    """

    plt.figure(figsize=(12, 6))
    sns.heatmap(probabilities, cmap='viridis', cbar=True)
    plt.xlabel('Cluster')
    plt.ylabel('Document')
    plt.title('Document-Cluster Probability Heatmap (GMM)')
    plt.show()


def cluster_probability_distribution(probabilities):
    """
    Create a bar plot to visualize the probability distribution of each cluster
    
    Parameters:
        probabilities_diag : ndarray
    """

    # For each cluster, show how many papers have different probability levels
    n_clusters = probabilities.shape[1]

    # Define probability bins
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
    bin_labels = ['<50%', '50-59%', '60-69%', '70-79%', '80-89%', '90-99%', '100%']

    # For each cluster, bin the probabilities
    cluster_distributions = []
    for cluster_id in range(n_clusters):
        cluster_probs = probabilities[:, cluster_id]
        # Bin the probabilities
        binned = pd.cut(cluster_probs, bins=bins, labels=bin_labels, include_lowest=True)
        counts = binned.value_counts().reindex(bin_labels, fill_value=0)
        # Calculate percentages
        percentages = (counts / len(cluster_probs)) * 100
        cluster_distributions.append(percentages)

    dist_df = pd.DataFrame(cluster_distributions, index=[f'Cluster {i}' for i in range(n_clusters)])

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']
    dist_df.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_xlabel('Cluster', fontsize=18)
    ax.set_ylabel('Percentage of Papers (%)', fontsize=18)
    ax.set_title('Distribution of Cluster Assignment Probabilities by Cluster (GMM)', fontsize=14)
    ax.legend(title='Probability Range', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_ylim(80, 100)
    plt.tight_layout()
    plt.show()

    # Print statistics for each cluster
    print("\n--- Cluster Assignment Certainty Statistics ---")
    for cluster_id in range(n_clusters):
        cluster_probs = probabilities[:, cluster_id]
        high_conf = (cluster_probs > 0.9).sum()
        medium_conf = ((cluster_probs > 0.7) & (cluster_probs <= 0.9)).sum()
        low_conf = ((cluster_probs > 0.5) & (cluster_probs <= 0.7)).sum()
        total = len(cluster_probs)
        print(f"Cluster {cluster_id}: >90%: {high_conf} ({high_conf/total*100:.1f}%), 70-90%: {medium_conf} ({medium_conf/total*100:.1f}%), 50-70%: {low_conf} ({low_conf/total*100:.1f}%)")