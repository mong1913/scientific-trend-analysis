import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

def plot_dim_reduction(embedding_2d, cluster_labels, technique_name, 
                       cluster_id, sample_size, figsize=(10, 7)):
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
    """
    # Create DataFrame with consistent column naming
    col_names = [f'{technique_name}1', f'{technique_name}2']
    df_plot = pd.DataFrame(embedding_2d, columns=col_names)
    df_plot[cluster_id] = cluster_labels
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=df_plot, 
        x=col_names[0], 
        y=col_names[1],
        hue=cluster_id,
        palette='tab10',
        s=15, 
        alpha=0.7
    )
    
    plt.title(f'{technique_name} Visualization with KMeans Clusters (stratified n={sample_size:,})', 
              fontsize=12, fontweight='bold')
    plt.xlabel(f'{technique_name} Dimension 1', fontsize=11)
    plt.ylabel(f'{technique_name} Dimension 2', fontsize=11)
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    print(f"{technique_name} visualization complete with {len(df_plot[cluster_id].unique())} clusters")
    
    return df_plot



def silhouette_plot(lsa_result, cluster_labels, best_k):
    """
    Create a silhouette plot to visualize clustering results
    
    Parameters:
        lsa_result : ndarray, shape (n_samples, n_components)
            Transformed data using LSA
        cluster_labels : array-like
            Cluster assignments for each sample
        best_k : int
            Best number of clusters determined by silhouette score
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

        colors = plt.cm.tab10(range(best_k))
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

    ax.set_title(f"Silhouette Plot (KMeans k={best_k}, sampled n={len(lsa_result):,})", 
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
