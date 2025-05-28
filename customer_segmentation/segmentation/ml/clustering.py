import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_clustering(data, algorithm, n_clusters=3, random_state=42):
    """
    Perform clustering on preprocessed data
    
    Args:
        data: preprocessed DataFrame
        algorithm: clustering algorithm to use ('kmeans', 'dbscan', etc.)
        n_clusters: number of clusters for algorithms that require this parameter
        random_state: random seed for reproducibility
        
    Returns:
        labels: cluster labels for each data point
        metrics: evaluation metrics
        cluster_profiles: summary statistics for each cluster
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Apply the selected clustering algorithm
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(data)
        
        # Calculate metrics
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette score
            metrics['silhouette_score'] = float(silhouette_score(data, labels))
        metrics['inertia'] = float(model.inertia_)
        
    elif algorithm == 'dbscan':
        # For DBSCAN, we need to determine eps and min_samples
        # For simplicity, we'll use default values
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(data)
        
        # Calculate metrics
        unique_labels = np.unique(labels)
        n_clusters_actual = len(unique_labels) - (1 if -1 in labels else 0)
        metrics['n_clusters'] = n_clusters_actual
        metrics['n_noise'] = list(labels).count(-1)
        
        if len(unique_labels) > 1 and -1 not in unique_labels:  # Only if no noise points
            metrics['silhouette_score'] = float(silhouette_score(data, labels))
    
    elif algorithm == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(data)
        
        # Calculate metrics
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(data, labels))
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Create profiles for each cluster
    cluster_profiles = create_cluster_profiles(data, labels)
    
    return labels, metrics, cluster_profiles

def create_cluster_profiles(data, labels):
    """
    Create profiles for each cluster with summary statistics
    
    Args:
        data: preprocessed DataFrame
        labels: cluster labels
        
    Returns:
        cluster_profiles: dictionary with cluster statistics
    """
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    # Get unique clusters
    unique_clusters = sorted(np.unique(labels))
    
    # Initialize profiles dictionary
    cluster_profiles = {}
    
    # Calculate statistics for each cluster
    for cluster_id in unique_clusters:
        # Skip noise points (DBSCAN assigns -1 to noise)
        if cluster_id == -1:
            continue
            
        # Get data for this cluster
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id].drop('cluster', axis=1)
        
        # Calculate statistics
        profile = {
            'size': len(cluster_data),
            'proportion': len(cluster_data) / len(data),
        }
        
        # Add mean values for each feature
        for column in cluster_data.columns:
            profile[f'mean_{column}'] = float(cluster_data[column].mean())
        
        # Add the profile to the dictionary
        cluster_profiles[str(cluster_id)] = profile
    
    return cluster_profiles