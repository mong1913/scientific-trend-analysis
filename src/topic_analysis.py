from collections import Counter
import pandas as pd
import numpy as np

def remove_redundant_terms(word_list, n_final=10):
    """
    Remove redundant terms where unigrams are contained in higher-scoring bigrams.
    
    Parameters:
    - word_list: list of words sorted by score descending
    - n_final: number of final unique terms to return
    
    Returns:
    - filtered list of words with redundancy removed
    """
    filtered = []
    bigrams = [word for word in word_list if ' ' in word]
    
    for word in word_list:
        if ' ' in word:
            filtered.append(word)
        else:
            is_redundant = any(word in bigram for bigram in bigrams)
            if not is_redundant:
                filtered.append(word)
        
        if len(filtered) >= n_final:
            break
    
    return filtered[:n_final]


def get_top_frequency_words_per_topic(df, cluster_labels, n_words=10, remove_redundancy=True):
    """
    Function to get the top n words for each topic based on word frequency.

    Parameters:
    - df: pandas DataFrame containing the original data
    - cluster_labels: string, column name containing the cluster labels (e.g., 'Cluster_ID_km')
    - n_words: number of top words to retrieve for each topic
    - remove_redundancy: if True, removes unigrams that appear in bigrams
    
    Returns:
    - topic_summary: dictionary mapping cluster_id to list of top words
    """

    text_column = 'processed_text'
    df_cluster_text = df.groupby(cluster_labels)[text_column].agg(lambda x: ' '.join(x))
    df_cluster_text = df_cluster_text.to_frame(name='Combined_Text')

    # Precompute document counts per cluster
    doc_counts = df.groupby(cluster_labels).size().to_dict()

    topic_summary = {}

    print("\n--- Top Words for Topic Interpretation (Word Frequency) ---")

    for cluster_id, row in df_cluster_text.iterrows():
        combined_text = row['Combined_Text']
        words = combined_text.split()
        word_counts = Counter(words)

        # Get more words initially for redundancy removal
        # Get top n words
        initial_n = n_words * 2 if remove_redundancy else n_words
        top_words = word_counts.most_common(initial_n)
        top_words_list = [item[0] for item in top_words]
        
        # Remove redundant unigrams if bigrams are present
        if remove_redundancy:
            top_words_list = remove_redundant_terms(top_words_list, n_words)
        else:
            top_words_list = top_words_list[:n_words]
        
        topic_summary[cluster_id] = top_words_list
        
        # Print number of documents and keywords in each cluster
        keywords = ', '.join(top_words_list)
        num_docs = doc_counts.get(cluster_id, 0)
        percentage = num_docs / len(df) * 100
        print(f"Cluster {cluster_id} (Documents: {num_docs}, {percentage:.2f}%): {keywords}")

    # return topic_summary


def get_top_tfidf_words_per_topic(df, X_tfidf_reduced, feature_names, cluster_labels, n_words=10, remove_redundancy=True):
    """
    Function to get the top n words for each topic based on TF-IDF scores.
    Uses the pre-computed TF-IDF matrix.
    
    Parameters:
    - df: pandas DataFrame containing the original data with cluster assignments
    - X_tfidf_reduced: sparse matrix, the TF-IDF matrix (after variance threshold)
    - feature_names: array, feature names from TF-IDF vectorizer
    - cluster_labels: string, column name containing the cluster labels (e.g., 'Cluster_ID_km')
    - n_words: number of top words to retrieve for each topic
    - remove_redundancy: if True, removes unigrams that appear in bigrams
    
    Returns:
    - topic_summary: dictionary mapping cluster_id to list of top words
    """
    
    # Get unique cluster IDs from the dataframe column
    unique_clusters = df[cluster_labels].unique()
    unique_clusters = sorted(unique_clusters)
    
    topic_summary = {}
    
    print("\n--- Top Words for Topic Interpretation (TF-IDF Scores) ---")
    
    for cluster_id in unique_clusters:
        # Get indices of documents in this cluster
        cluster_mask = (df[cluster_labels] == cluster_id).values
        
        # Extract only the rows for this cluster from sparse matrix
        cluster_tfidf = X_tfidf_reduced[cluster_mask]
        cluster_tfidf_dense = cluster_tfidf.toarray()
        
        # Average TF-IDF scores across all documents in the cluster
        avg_tfidf_scores = cluster_tfidf_dense.mean(axis=0)
        
        # Get top n words
        initial_n = n_words * 2 if remove_redundancy else n_words
        top_indices = np.argsort(avg_tfidf_scores)[::-1][:initial_n]
        top_words = [feature_names[i] for i in top_indices]
        
        # Remove redundant unigrams if bigrams are present
        if remove_redundancy:
            top_words = remove_redundant_terms(top_words, n_words)
        else:
            top_words = top_words[:n_words]
        
        topic_summary[cluster_id] = top_words
        
        # Format output
        keywords = ', '.join([f"{word}" for word in top_words])
        num_docs = cluster_mask.sum()
        percentage = num_docs / len(df) * 100
        print(f"Cluster {cluster_id} (Documents: {num_docs}, {percentage:.2f}%): {keywords}")
    
    # return topic_summary
