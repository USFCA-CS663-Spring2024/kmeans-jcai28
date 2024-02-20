import numpy as np
class cluster:

    def __init__(self, k=5, max_iterations=100):
        # Allow the class’ users to set the algorithm’s hyperparameters
        # The default values are required to be k = 5 and max_iterations = 100.
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X, balanced = False):
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

        if balanced:
            cluster_hypotheses, centroids = self.balance_clusters(X, cluster_hypotheses, centroids)
        # Return A. A list (of length n) of the cluster hypotheses, one for each instance.
        # Return B. A list (of length at most k) containing lists (each of length d) of the cluster centroids’ values.
        return cluster_hypotheses, centroids
    

    def balance_clusters(self, X, cluster_hypotheses, centroids):
        # Count the number of instances in each cluster
        cluster_counts = [cluster_hypotheses.count(i) for i in range(self.k)]
        print("Original cluster: ",cluster_counts)

        # Calculate the target number of instances for each cluster
        target_cluster_count = sum(cluster_counts)//len(cluster_counts)

        small_clusters = []

        # Redistribute the points among the clusters
        for i in range(self.k):
            if cluster_counts[i] > target_cluster_count:
                # Remove points from large clusters
                instances_to_remove = cluster_counts[i] - target_cluster_count
                cluster_indices = [j for j in range(len(cluster_hypotheses)) if cluster_hypotheses[j] == i]
                distances_to_centroid = [np.linalg.norm(np.array(X[j]) - np.array(centroids[i])) for j in cluster_indices]
                instances_to_remove_indices = np.argsort(distances_to_centroid)[-instances_to_remove:]
                for index in instances_to_remove_indices:
                    cluster_hypotheses[cluster_indices[index]] = -1  # Remove the point from its current cluster assignment
            if cluster_counts[i] < target_cluster_count:
                # Find small clusters
                small_clusters.append(i)

        cluster_counts = [cluster_hypotheses.count(i) for i in range(self.k)]
        print("Remove the out points for the large cluster: ",cluster_counts)

        # Redistribute the removed points among the clusters based on their distances to the centroids
        for i in range(len(cluster_hypotheses)):
            if cluster_hypotheses[i] == -1:
                distances = [np.linalg.norm(np.array(X[i]) - np.array(centroids[j])) for j in small_clusters]
                cluster_hypotheses[i] = small_clusters[np.argmin(distances)]

        cluster_counts = [cluster_hypotheses.count(i) for i in range(self.k)]
        print("Redistribute the removed points to the small cluster: ",cluster_counts)


        # Recalculate the centroids
        new_centroids = []
        for i in range(self.k):
            cluster_indices = [j for j in range(len(cluster_hypotheses)) if cluster_hypotheses[j] == i]
            cluster_points = [X[j] for j in cluster_indices]
            if cluster_points:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[i])

        return cluster_hypotheses, new_centroids


