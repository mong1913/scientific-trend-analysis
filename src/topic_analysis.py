from collections import Counter

def get_top_words_per_topic(df, cluster_labels, n_words):
    """
    Function to get the top n words for each topic.

    Parameters:
    - df: pandas DataFrame containing the original data
    - cluster_labels: numpy array containing the cluster labels for each data point
    - df_cluster_text: pandas DataFrame containing the combined text for each cluster
    - n_words: number of top words to retrieve for each topic
    """

    text_column = 'processed_text'
    df_cluster_text = df.groupby(cluster_labels)[text_column].agg(lambda x: ' '.join(x))
    df_cluster_text = df_cluster_text.to_frame(name='Combined_Text')

    top_n_words = n_words 
    topic_summary = {}

    print("\n--- Top Words for Topic Interpretation ---")

    for cluster_id, row in df_cluster_text.iterrows():
        combined_text = row['Combined_Text']
        words = combined_text.split()
        word_counts = Counter(words)
        top_words = word_counts.most_common(top_n_words)
        
        top_words_list = [item[0] for item in top_words]
        topic_summary[cluster_id] = top_words_list
        
        # Combined output on single line
        keywords = ', '.join(top_words_list)
        print(f"Cluster {cluster_id} (Documents: {len(words)} words): {keywords}")