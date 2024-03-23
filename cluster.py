from collections import defaultdict
from numpy.typing import NDArray
from typing import List, Tuple, Dict
import numpy as np

# from control import grid_2_cluster

gravity_constant_3d = np.sqrt(2 * np.pi ** 3)
k_array = np.asarray([10, 10, 1])
epsilon = 10 ** -6
cluster_2_gird_dict = Dict[int, List[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
grid_2_cluster_dict = Dict[Tuple[int, int], int]


class Cluster:
    clusters = defaultdict()

    def __init__(self, index: int, members_grid: List[Tuple[int, int]]):
        self.index = index
        self.members_grid = members_grid
        self.clusters[index] = self
        self.data = None
        self.mean = None
        self.gravity = None
        self.inv_cov_matrix = None
        self.std = None

    def remove(self):
        self.clusters.pop(self.index)


def initiate_clustering(cluster_2_grid: cluster_2_gird_dict) -> None:
    for index in cluster_2_grid.keys():
        Cluster(index=index, members_grid=cluster_2_grid[index])


def update_all_clusters(grid_2_data: grid_2_data_dict, k_adjustments: NDArray) -> None:
    cluster_keys_copy = list(Cluster.clusters.keys())
    for index in cluster_keys_copy:
        update_cluster(index, grid_2_data, k_adjustments)


def update_cluster(cluster_index: int, grid_2_data: grid_2_data_dict, k_adjustments: NDArray) -> None:
    cluster = Cluster.clusters[cluster_index]
    if len(cluster.members_grid) <= 1:
        cluster.remove()
        return
    update_cluster_data(grid_2_data, cluster)
    update_cluster_mean(cluster)
    update_cluster_std(cluster)
    update_cluster_gravity(cluster, k_adjustments)


def update_cluster_data(grid_2_data: grid_2_data_dict, cluster: Cluster) -> None:
    try:
        cluster_arr = np.vstack([grid_2_data[grid] for grid in cluster.members_grid])
        cluster.data = cluster_arr
    except ValueError:
        ...


def update_cluster_mean(cluster: Cluster) -> None:
    cluster.mean = np.mean(cluster.data, axis=0)


def update_cluster_std(cluster: Cluster) -> None:
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
    cluster.gravity = gravity_constant_3d * det * force * np.exp(force / 2)
