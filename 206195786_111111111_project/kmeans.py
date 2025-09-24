import math
import sys
EPSILON = 0.001

def read_vectors_from_stdin():
    vectors = []
    for line in sys.stdin:
        if line.strip():  # Skip empty lines if any
            vector = list(map(float, line.strip().split(','))) # strip the line, seperate it by commmas, converts it to floats and the list of floats
            vectors.append(vector)
    return vectors

def read_vectors(file_path):
    vectors = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines if any
                vector = list(map(float, line.strip().split(','))) # strip the line, seperate it by commmas, converts it to floats and the list of floats
                vectors.append(vector)
    return vectors

def initialize_centroids(vectors, k):
    return vectors[:k]

def euclidean_distance(point1, point2):
    distance_sum = 0.0
    for i in range(len(point1)):
        distance_sum += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance_sum)

def assign_points_to_centroids(vectors, centroids):
    assignments = []
    
    for vector in vectors:
        min_distance = None
        assigned_centroid = None
        
        for idx in range(len(centroids)):
            distance = euclidean_distance(vector, centroids[idx])
            
            if (min_distance is None) or (distance < min_distance):
                min_distance = distance
                assigned_centroid = idx
                
        assignments.append(assigned_centroid)
    
    return assignments

def update_centroids(vectors, assignments, centroids):
    k = len(centroids)
    clusters = [[] for _ in range(k)]
    
def update_centroids(vectors, assignments, centroids):
    k = len(centroids)
    clusters = [[] for _ in range(k)]
    
    for idx, vector in enumerate(vectors):
        cluster_index = assignments[idx]
        clusters[cluster_index].append(vector)
    
    new_centroids = []
    for idx in range(k):
        cluster = clusters[idx]
        if cluster:
            centroid = []
            dimensions = len(cluster[0])
            for dim in range(dimensions):
                coord_sum = sum(point[dim] for point in cluster)
                centroid.append(coord_sum / len(cluster))
            new_centroids.append(centroid)
        else:
            # Keep previous centroid if no points assigned
            new_centroids.append(centroids[idx])
    
    return new_centroids

def k_means(vectors, k, max_iterations=None):
    if max_iterations is None:
        max_iterations = 400

    centroids = initialize_centroids(vectors,k)
    
    for _ in range(max_iterations):
        assignments = assign_points_to_centroids(vectors, centroids)
        new_centroids = update_centroids(vectors, assignments, centroids)
        
        # Check if ALL centroids moved less than EPSILON
        converged = True
        for i in range(k):
            if euclidean_distance(centroids[i], new_centroids[i]) >= EPSILON:
                converged = False
                break
        
        if converged:
            break
        
        centroids = new_centroids

    return centroids


# Formats centroids to 4 decimal places as CSV lines
def format_output(centroids):
    formatted = []
    for centroid in centroids:
        line = ','.join(f'{coord:.4f}' for coord in centroid)
        formatted.append(line)
    return formatted

def is_valid_k(k, N):
    return 1 < k < N

def is_valid_iter(max_iter):
    return 1 < max_iter < 1000

def main():
    try:
        args = sys.argv
        if len(args) not in [2, 3]: print("An Error Has Occurred"); sys.exit(1)
        try: k_float = float(args[1])
        except ValueError: print("Incorrect number of clusters!"); sys.exit(1)
        if not k_float.is_integer(): print("Incorrect number of clusters!"); sys.exit(1)
        k = int(k_float)
        if len(args) == 3:
            try: iter_float = float(args[2])
            except ValueError: print("Incorrect maximum iteration!"); sys.exit(1)
            if not iter_float.is_integer(): print("Incorrect maximum iteration!"); sys.exit(1)
            max_iter = int(iter_float)
        else: max_iter = 400
        if sys.stdin.isatty(): print("An Error Has Occurred"); sys.exit(1)
        vectors = read_vectors_from_stdin()
        N = len(vectors)
        if not is_valid_k(k, N): print("Incorrect number of clusters!"); sys.exit(1)
        if not is_valid_iter(max_iter): print("Incorrect maximum iteration!"); sys.exit(1)
        centroids = k_means(vectors, k, max_iter)
        for line in format_output(centroids): print(line)
    except Exception: print("An Error Has Occurred"); sys.exit(1)

if __name__ == "__main__":
    main()
