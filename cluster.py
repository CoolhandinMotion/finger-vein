from collections import defaultdict
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Set
import numpy as np

GRAVITY_CONSTANT_3D = np.sqrt(2 * np.pi ** 3)

cluster_2_gird_dict = Dict[int, Set[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
grid_2_cluster_dict = Dict[Tuple[int, int], int]



class Cluster:

    def __init__(self, index: int, members_grid: Set[Tuple[int, int]]):
        self.index = index
        self.members_grid = members_grid
        self.data = None
        self.mean = None
        self.gravity = None
        self.inv_cov_matrix = None
        self.std = None


def initiate_clustering(cluster_2_grid: cluster_2_gird_dict, clusters: Dict[int, Cluster]) -> None:
    for index in cluster_2_grid.keys():
        clusters[index] = Cluster(index=index, members_grid=cluster_2_grid[index])


def update_all_clusters(grid_2_data: grid_2_data_dict,
                        k_adjustments: NDArray, clusters: Dict[int, Cluster],epsilon: float) -> None:
    cluster_keys_copy = list(clusters.keys())
    for index in cluster_keys_copy:
        update_cluster(index, grid_2_data, k_adjustments, clusters, epsilon)


def update_cluster(cluster_index: int, grid_2_data: grid_2_data_dict,
                   k_adjustments: NDArray, clusters: Dict[int, Cluster],epsilon:float) -> None:
    cluster = clusters[cluster_index]
    if len(cluster.members_grid) <= 1:
        clusters.pop(cluster_index)
        return
    update_cluster_data(grid_2_data, cluster)
    update_cluster_mean(cluster)
    update_cluster_std(cluster,epsilon=epsilon)
    update_cluster_gravity(cluster, k_adjustments)


def update_cluster_data(grid_2_data: grid_2_data_dict, cluster: Cluster) -> None:
    try:
        cluster_arr = np.vstack([grid_2_data[grid] for grid in cluster.members_grid])
        cluster.data = cluster_arr
    except ValueError:
        ...


def update_cluster_mean(cluster: Cluster) -> None:
    cluster.mean = np.mean(cluster.data, axis=0)


def update_cluster_std(cluster: Cluster, epsilon: float) -> None:
    std = np.std(cluster.data, axis=0)
    # below should be a one liner lambda function
    if std.any() == 0:
        for i in range(3):
            if std[i] == 0:
                std[i] = epsilon
    cluster.std = std


def update_cluster_gravity(cluster: Cluster, k_adjustments: NDArray) -> None:
    k_times_sigma = cluster.std * k_adjustments
    adjusted_force_array = (np.power(k_times_sigma + 1, -1) * k_adjustments) ** 2
    force = np.sum(adjusted_force_array)
    det = np.product(k_times_sigma)
    cluster.gravity = GRAVITY_CONSTANT_3D * det * force * np.exp(force / 2)
