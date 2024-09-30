import numpy as np
import random

class KMeans:
    def __init__(self, k, init_method="random"):
        self.k = k
        self.init_method = init_method
        self.centroids = None
        self.data = None
        self.converged = False

    def initialize_centroids(self, data):
        self.data = data
        if self.init_method == "random":
            self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        elif self.init_method == "farthest_first":
            self.centroids = self.farthest_first_initialization(data)
        elif self.init_method == "kmeans++":
            self.centroids = self.kmeans_plus_plus_initialization(data)
        # For manual selection, centroids will be set dynamically.

    def farthest_first_initialization(self, data):
        centroids = [random.choice(data)]
        for _ in range(1, self.k):
            dist = np.min([np.linalg.norm(data - c, axis=1) for c in centroids], axis=0)
            next_centroid = data[np.argmax(dist)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, data):
        centroids = [data[np.random.choice(len(data))]]
        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in data])
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = random.random()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(data[j])
                    break
        return np.array(centroids)


    def assign_clusters(self):
        clusters = []
        for point in self.data:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def update_centroids(self, clusters):
        new_centroids = []
        for i in range(self.k):
            points_in_cluster = self.data[clusters == i]
            if points_in_cluster.size > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
        self.centroids = np.array(new_centroids)

    def iterate(self):
        if self.converged:
            return self.clusters, self.converged

        prev_centroids = self.centroids.copy()
        self.clusters = self.assign_clusters()
        self.update_centroids(self.clusters)

        # Check if centroids have changed
        if np.all(prev_centroids == self.centroids):
            self.converged = True
        
        return self.clusters, self.converged
