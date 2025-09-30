import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict
from pathlib import Path
import os

# Define the range of K values to test
K_RANGE = range(1, 11)

def calculate_wcss_and_silhouette(data_2d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Within-Cluster Sum of Squares (WCSS/Inertia) and Silhouette Scores
    for a range of K values (clusters).

    Args:
        data_2d (pd.DataFrame): DataFrame containing the features for clustering.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. WCSS data (Columns: 'Clusters', 'WCSS')
            2. Silhouette data (Columns: 'Clusters', 'Silhouette Score')
    """
    print(f"\n--- Calculating WCSS and Silhouette Scores for K={min(K_RANGE)} to {max(K_RANGE)} ---")
    
    wcss = []
    sil_score = []
    
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=2, n_init='auto')
        kmeans.fit(data_2d)
        
        # 1. Calculate WCSS/Inertia (for all K values)
        wcss.append(kmeans.inertia_)
        
        # 2. Calculate Silhouette Score (only for K > 1)
        if k > 1:
            cluster_labels = kmeans.predict(data_2d)
            score = silhouette_score(data_2d, cluster_labels)
            sil_score.append(score)
            
    # Create WCSS DataFrame (includes K=1)
    wcss_data = pd.DataFrame({'Clusters': list(K_RANGE), 'WCSS': wcss})
    print("\nWCSS Data:")
    print(wcss_data)
    
    # Create Silhouette DataFrame (excludes K=1)
    sil_data = pd.DataFrame({'Clusters': list(K_RANGE)[1:], 'Silhouette Score': sil_score})
    print("\nSilhouette Score Data:")
    print(sil_data)
    
    return wcss_data, sil_data

def plot_cluster_evaluation(wcss_data: pd.DataFrame, sil_data: pd.DataFrame):
    """
    Generates and saves plots for WCSS (Elbow Method) and Silhouette Scores.

    Args:
        wcss_data (pd.DataFrame): DataFrame containing WCSS values.
        sil_data (pd.DataFrame): DataFrame containing Silhouette Scores.
    """
    # 1. Ensure the 'graphs' directory exists
    graphs_dir = 'graphs'
    Path(graphs_dir).mkdir(parents=True, exist_ok=True)
    
    # --- Plot 1: WCSS (Elbow Method) ---
    plt.figure(figsize=(10, 6))
    plt.plot(wcss_data['Clusters'], wcss_data['WCSS'], marker='o', linestyle='-', color='blue')
    plt.title('Elbow Method: WCSS vs. Number of Clusters (K)', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('WCSS (Inertia)', fontsize=12)
    plt.xticks(wcss_data['Clusters'])
    plt.grid(True, linestyle='--', alpha=0.6)
    wcss_path = os.path.join(graphs_dir, 'elbow_method_wcss.png')
    plt.savefig(wcss_path)
    plt.close()
    print(f"WCSS plot saved to '{wcss_path}'")

    # --- Plot 2: Silhouette Scores ---
    plt.figure(figsize=(10, 6))
    plt.plot(sil_data['Clusters'], sil_data['Silhouette Score'], marker='o', linestyle='-', color='red')
    plt.title('Silhouette Score vs. Number of Clusters (K)', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    plt.xticks(sil_data['Clusters'])
    plt.grid(True, linestyle='--', alpha=0.6)
    sil_path = os.path.join(graphs_dir, 'silhouette_scores.png')
    plt.savefig(sil_path)
    plt.close()
    print(f"Silhouette Score plot saved to '{sil_path}'")
