import itertools
import random
from typing import List, NamedTuple, Union, Callable, Tuple

import tqdm

from dsfs.linalg.vector import Vector, vector_mean, squared_distance, distance


def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


def cluster_means(k: int, inputs: List[Vector], assignments: List[int]) -> List[Vector]:
    clusters = [[] for i in range(k)]
    for inp, assignment in zip(inputs, assignments):
        clusters[assignment].append(inp)

    return [vector_mean(cluster) if cluster else random.choice(inputs) for cluster in clusters]


class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k
        self.means = None

    def classify(self, inp: Vector) -> int:
        return min(range(self.k), key=lambda i: squared_distance(inp, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(inp) for inp in inputs]

                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")


def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = [clusterer.classify(inp) for inp in inputs]

    return sum(squared_distance(inp, means[cluster]) for inp, cluster in zip(inputs, assignments))


class Leaf(NamedTuple):
    value: Vector


class Merged(NamedTuple):
    children: tuple
    order: int


Cluster = Union[Leaf, Merged]


def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value for child in cluster.children for value in get_values(child)]


def clsuter_distance(cluster1: Cluster, cluster2: Cluster, distance_agg: Callable = min) -> float:
    return distance_agg(
        [distance(v1, v2) for v1 in get_values(cluster1) for v2 in get_values(cluster2)]
    )


def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float("inf")
    else:
        return cluster.order


def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children


def bottom_up_cluster(inputs: List[Vector], distance_agg: Callable = min) -> Cluster:
    clusters: List[Cluster] = [Leaf(inp) for inp in inputs]

    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        return clsuter_distance(pair[0], pair[1], distance_agg)

    while len(clusters) > 1:
        c1, c2 = min(
            (
                (cluster1, cluster2)
                for i, cluster1 in enumerate(clusters)
                for cluster2 in clusters[:i]
            ),
            key=pair_distance,
        )

        clusters = [c for c in clusters if c != c1 and c != c2]

        merged_cluster = Merged((c1, c2), order=len(clusters))

        clusters.append(merged_cluster)

    return clusters[0]


def generate_clusters(base_cluster: Cluster, num_clusters: int) -> List[Cluster]:
    clusters = [base_cluster]

    while len(clusters) < num_clusters:
        next_cluster = min(clusters, key=get_merge_order)
        clusters = [c for c in clusters if c != next_cluster]
        clusters.extend(get_children(next_cluster))

    return clusters
