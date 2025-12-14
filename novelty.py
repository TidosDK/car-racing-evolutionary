import numpy as np

class NoveltyArchive:
    def __init__(self, threshold=20.0, k_neighbors=15, limit=2500):
        """
        :param threshold: Minimum sparseness required to be added to the archive.
        :param k_neighbors: Number of nearest neighbors to consider for calculation.
        :param limit: Maximum size of the archive to prevent memory issues.
        """
        self.archive = []
        self.threshold = threshold
        self.k_neighbors = k_neighbors
        self.limit = limit

    def calculate_novelty(self, behavior, population_behaviors):
        """
        Calculates the average distance to the k-nearest neighbors
        in both the archive and the current population.
        """
        all_behaviors = self.archive + population_behaviors

        target = np.array(behavior)

        distances = [np.linalg.norm(target - np.array(b)) for b in all_behaviors]

        if not distances:
            return 0.0

        distances.sort()

        k_nearest = distances[:self.k_neighbors]

        if not k_nearest:
            return 0.0

        sparseness = sum(k_nearest) / len(k_nearest)
        return sparseness

    def update_archive(self, behavior, sparseness):
        """
        Adds the behavior to the archive if it is novel enough.
        """
        if sparseness > self.threshold or len(self.archive) < 10:
            self.archive.append(behavior)

            # If archive grows too big, remove a random element (standard NS practice)
            if len(self.archive) > self.limit:
                self.archive.pop(np.random.randint(0, len(self.archive)))

    def size(self):
        return len(self.archive)
