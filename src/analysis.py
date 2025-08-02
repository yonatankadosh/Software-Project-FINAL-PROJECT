import sys
import numpy as np
import symnmf
from kmeans import k_means, read_vectors

# Set random seed for reproducible results
np.random.seed(1234)

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def silhouette_score_manual(X, labels):
    """
    Calculate silhouette score manually.
    
    Silhouette coefficient = (b - a) / max(a, b)
    Where:
    - a = mean distance between point and all other points in same cluster
    - b = minimum mean distance between point and all points in other clusters
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n_samples):
        point = X[i]
        cluster_label = labels[i]
        
        # Calculate a: mean distance to points in same cluster
        same_cluster_points = X[labels == cluster_label]
        if len(same_cluster_points) == 1:  # Only this point in cluster
            a = 0.0
        else:
            distances = [euclidean_distance(point, other_point) for other_point in same_cluster_points if not np.array_equal(point, other_point)]
            a = np.mean(distances) if distances else 0.0
        
        # Calculate b: minimum mean distance to points in other clusters
        b_values = []
        for other_label in unique_labels:
            if other_label != cluster_label:
                other_cluster_points = X[labels == other_label]
                distances = [euclidean_distance(point, other_point) for other_point in other_cluster_points]
                b_values.append(np.mean(distances))
        
        if not b_values:  # Only one cluster
            b = 0.0
        else:
            b = min(b_values)
        
        # Calculate silhouette coefficient for this point
        if max(a, b) == 0:
            silhouette_coef = 0.0
        else:
            silhouette_coef = (b - a) / max(a, b)
        
        silhouette_scores.append(silhouette_coef)
    
    # Return mean silhouette score
    return np.mean(silhouette_scores)

def get_symnmf_clusters(X, k):
    """
    Apply SymNMF clustering and return cluster assignments.
    For SymNMF, cluster assignment is done by assigning each point to the cluster
    corresponding to the maximum value in the H matrix row.
    """
    # Compute similarity matrix
    A = symnmf.compute_similarity_matrix(X)
    D = symnmf.compute_diagonal_degree_matrix(A)
    W = symnmf.compute_normalized_similarity_matrix(A, D)
    
    # Initialize H matrix
    H_init = np.random.uniform(0, 2 * np.sqrt(1 / k), (X.shape[0], k))
    
    # Run SymNMF
    H_final = symnmf.symnmf(W, H_init, 300, 1e-4)
    
    # Assign clusters based on maximum value in each row of H
    clusters = np.argmax(H_final, axis=1)
    return clusters

def get_kmeans_clusters(X, k):
    """
    Apply K-means clustering and return cluster assignments.
    """
    # Convert numpy array to list of lists for kmeans implementation
    vectors = X.tolist()
    
    # Run K-means with correct convergence parameters: max_iter=300, epsilon=1e-4
    centroids = k_means(vectors, k, 300)
    
    # Assign points to nearest centroids
    from kmeans import assign_points_to_centroids
    clusters = assign_points_to_centroids(vectors, centroids)
    return np.array(clusters)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analysis.py <k> <file_name.txt>")
        return
    
    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
        
        # Read data
        X = np.loadtxt(file_name, delimiter=',')
        
        # Apply SymNMF clustering
        symnmf_clusters = get_symnmf_clusters(X, k)
        symnmf_score = silhouette_score_manual(X, symnmf_clusters)
        
        # Apply K-means clustering
        kmeans_clusters = get_kmeans_clusters(X, k)
        kmeans_score = silhouette_score_manual(X, kmeans_clusters)
        
        # Print results
        print(f"nmf: {symnmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")
        
    except Exception as e:
        print(f"An Error Has Occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
