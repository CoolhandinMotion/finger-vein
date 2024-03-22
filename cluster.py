from collections import defaultdict
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import List, Tuple, Dict
import numpy as np

gravity_constant_3d = np.sqrt(2*np.pi**3)
k_array = np.asarray([10,10,1])
epsilon = 10**-6
cluster_2_gird_dict = Dict[int, List[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
def initiate_clustering(cluster_2_grid: cluster_2_gird_dict) -> None:
    for index in cluster_2_grid.keys():
         c = Cluster(index = index, members_grid= cluster_2_grid[index])
def get_cluster_data_from_grid(grid_2_data: Dict[Tuple[int, int], NDArray],
                               members_grid:List[Tuple[int, int]]) -> NDArray:

    cluster_arr = np.vstack([grid_2_data[grid] for grid in members_grid])
    return cluster_arr

def update_cluster(cluster_index: int,grid_2_data: grid_2_data_dict, k_adjusments: NDArray)->None:
    update_cluster_data(grid_2_data,cluster_index)
    update_cluster_mean(cluster_index)
    update_cluster_std(cluster_index)
    update_cluster_gravity(cluster_index,k_adjusments)

def update_cluster_data(grid_2_data: grid_2_data_dict,cluster_index: int) -> None:
    cluster = Cluster.clusters[cluster_index]
    cluster.data = get_cluster_data_from_grid(grid_2_data,cluster.members_grid)
def update_cluster_mean(cluster_index: int) -> None:
    cluster = Cluster.clusters[cluster_index]
    cluster.mean = np.mean(cluster.data,axis=0)
def update_cluster_std(cluster_index: int) ->None:
    cluster = Cluster.clusters[cluster_index]
    std = np.std(cluster.data,axis=0)
    # below should be a one liner lambda function
    if std.any() == 0:
        for i in range(3):
            if std[i] ==0:
                std[i] = epsilon
    cluster.std = std

def update_cluster_gravity(cluster_index: int, k_adjusments: NDArray) ->None:
    cluster = Cluster.clusters[cluster_index]
    k_times_sigma = cluster.std * k_adjusments
    adjusted_force_array = (np.power(k_times_sigma+1,-1)*k_adjusments)**2
    force = np.sum(adjusted_force_array)
    det = np.product(k_times_sigma)
    cluster.gravity = gravity_constant_3d*det*force*np.exp(force/2)







@dataclass
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






