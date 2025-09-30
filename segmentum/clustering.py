import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
from pathlib import Path
import os

# Define the target features for 2D clustering
TWO_D_FEATURES = ['Annual Income', 'Spending Score']
# Define the number of clusters
N_CLUSTERS = 5
# Define the colors for plotting the clusters
COLOR_DICT = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'yellow',
    4: 'magenta'
}

def prepare_2d_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the two primary features for 2D clustering into a new DataFrame.

    Args:
        df (pd.DataFrame): The main preprocessed DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing only 'Annual Income' and 'Spending Score'.
    """
    print(f"\n--- Preparing 2D Data for K-Means on {', '.join(TWO_D_FEATURES)} ---")
    data_2d = df[TWO_D_FEATURES].copy()
    print("2D Data Head:")
    print(data_2d.head())
    return data_2d

def run_kmeans_2d(data_2d: pd.DataFrame) -> Tuple[pd.DataFrame, KMeans]:
    """
    Performs K-Means clustering on the 2D data.

    Args:
        data_2d (pd.DataFrame): DataFrame with features for clustering.

    Returns:
        Tuple[pd.DataFrame, KMeans]: A tuple containing:
            1. data_2d DataFrame with a new 'Label' column.
            2. The fitted KMeans model object.
    """
    print(f"\n--- Running K-Means Clustering (K={N_CLUSTERS}) ---")
    
    # Initialize and fit the model
    # Set n_init='auto' to silence UserWarning in recent sklearn versions
    kmeans_2d = KMeans(n_clusters=N_CLUSTERS, random_state=2, n_init='auto') 
    kmeans_2d.fit(data_2d)
    
    # Predict labels and add to DataFrame
    labels_2d = kmeans_2d.predict(data_2d)
    data_2d = data_2d.copy() # Ensure we operate on a copy
    data_2d['Label'] = labels_2d
    
    print("2D Data Head with Labels:")
    print(data_2d.head())
    return data_2d, kmeans_2d

def plot_2d_clusters(data_2d: pd.DataFrame, kmeans_model: KMeans, save_path: str = 'graphs/2d_clusters.png'):
    """
    Generates a scatter plot of the clustered data, showing centroids and colors,
    and saves the plot to the specified path.

    Args:
        data_2d (pd.DataFrame): DataFrame with 'Annual Income', 'Spending Score', and 'Label'.
        kmeans_model (KMeans): The fitted K-Means model.
        save_path (str): The path and filename to save the plot (default: graphs/2d_clusters.png).
    """
    # 1. Ensure the directory exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    x_col, y_col = TWO_D_FEATURES
    
    # Determine the color for each data point
    color_list = [COLOR_DICT[label] for label in data_2d['Label']]
    
    # Start plotting
    plt.figure(figsize=(10, 6))
    
    # 1. Plot the actual data points with cluster colors
    scatter = plt.scatter(data_2d[x_col], data_2d[y_col], c=color_list, s=50, alpha=0.6, edgecolors='w', linewidths=0.5)
    
    # 2. Plot the cluster centroids (K-Means centers)
    centers = kmeans_model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], 
                marker='*', s=300, c='black', label='Centroids', edgecolors='white', linewidths=1.5)
    
    # Add title and labels
    plt.title(f'Customer Segments (K={N_CLUSTERS}): {x_col} vs {y_col}', fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    
    # Create custom legend for colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {k} ({COLOR_DICT[k]})', 
                          markerfacecolor=COLOR_DICT[k], markersize=10) for k in COLOR_DICT.keys()]
    plt.legend(handles=handles, title="Clusters", loc='lower right')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure instead of showing it
    plt.savefig(save_path)
    plt.close() # Close the figure to free up memory

def get_customers_by_cluster(main_df: pd.DataFrame, data_2d_labeled: pd.DataFrame, cluster_color: str = 'green') -> List[str]:
    """
    Identifies CustomerIDs belonging to a specific cluster (by color).

    Args:
        main_df (pd.DataFrame): The main DataFrame containing 'CustomerID'.
        data_2d_labeled (pd.DataFrame): The 2D DataFrame with the 'Label' column.
        cluster_color (str): The color corresponding to the target cluster (e.g., 'green').

    Returns:
        List[str]: A list of Customer IDs belonging to the specified cluster.
    """
    # Find the label index corresponding to the cluster color
    target_label = next(
        (label for label, color in COLOR_DICT.items() if color == cluster_color), 
        None
    )
    
    if target_label is None:
        print(f"Error: Cluster color '{cluster_color}' not found in COLOR_DICT.")
        return []

    print(f"\n--- Identifying Customers in the '{cluster_color}' cluster (Label {target_label}) ---")
    
    # Use the index alignment to filter the main DataFrame based on the 2D label
    target_indices = data_2d_labeled[data_2d_labeled['Label'] == target_label].index
    
    # Filter the main DataFrame using these indices
    cust_target = main_df.loc[target_indices]
    
    customer_ids = list(cust_target['CustomerID'])
    
    print(f"Total customers in '{cluster_color}' cluster: {len(customer_ids)}")
    print("Example Customer IDs:", customer_ids[:5], "...")
    
    return customer_ids