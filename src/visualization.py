import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
