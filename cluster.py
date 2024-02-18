import numpy as np
class cluster:

    def __init__(self, k=5, max_iterations=100):
        # Allow the class’ users to set the algorithm’s hyperparameters
        # The default values are required to be k = 5 and max_iterations = 100.
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # X is a list (not columns of a Dataframe) of n instances in d dimensions (features)
       
        n = len(X)  # number of instances
        d = len(X[0])  # number of dimensions/features
        
        # Randomly initialize k centroids
        centroids_indices = np.random.choice(n, self.k, replace=False)
        centroids = [X[i] for i in centroids_indices]

        # Initialize cluster hypotheses
        cluster_hypotheses = [-1] * n 

        iteration = 0
        while iteration < self.max_iterations:
            old_centroids = centroids.copy()
            
            # Assign each instance to the closest centroid
            for i in range(n):
                # The np.linalg.norm function compute the Euclidean distance 
                distances = [np.linalg.norm(np.array(X[i]) - np.array(centroid)) for centroid in centroids]
                # The argmin fucntion give the index of the min value in an array 
                cluster_hypotheses[i] = np.argmin(distances)

            # Update centroids
            for j in range(self.k):
                cluster_points = [X[i] for i in range(n) if cluster_hypotheses[i] == j]
                if cluster_points:
                    centroids[j] = np.mean(cluster_points, axis=0)

            # Check for convergence
            if np.array_equal(old_centroids, centroids):
                break
            
            iteration += 1
        
        # Return A. A list (of length n) of the cluster hypotheses, one for each instance.
        # Return B. A list (of length at most k) containing lists (each of length d) of the cluster centroids’ values.
        return cluster_hypotheses, centroids

