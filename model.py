from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set
from preprocessing import grid_tiling, load_image, ordinary_grid_2_data
from numpy._typing import NDArray
from cluster import Cluster
from collections import OrderedDict
import numpy as np


cluster_2_grid_dict = Dict[int, Set[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
grid_2_cluster_dict = Dict[Tuple[int, int], int]

GRAVITY_CONSTANT_3D = np.sqrt(2 * np.pi ** 3)


@dataclass(slots=True)
class Model:
    # def __init__(self, image_url: str,
    #              tile_size: int,
    #              k_adjustments: NDArray,
    #              epsilon: float = 10 ** -6,
    #              scale_image: bool = False):
    n_columns: int = field(init=False)
    n_rows: int = field(init=False)
    image_url: str
    tile_size: int
    k_adjustments: NDArray
    epsilon: float = 10 ** -6
    scale_image: bool = False
    grid_2_data: Dict[Tuple[int, int], NDArray] = field(init=False, default_factory=dict)
    grid_2_cluster: Dict[Tuple[int, int], int] = field(init=False, default_factory=dict)
    clusters: Dict[int, Cluster] = field(init=False, default_factory=OrderedDict)

    def __post_init__(self):
        self.__initiate_model()

    def __initiate_model(self) -> None:
        loaded_image = load_image(self.image_url, self.scale_image, round_decimal=3)
        self.n_rows, self.n_columns = loaded_image.shape[0], loaded_image.shape[1]
        self.grid_2_data = ordinary_grid_2_data(image=loaded_image)
        cluster_2_grid, self.grid_2_cluster = grid_tiling(grid_shape=loaded_image.shape, tile_size=self.tile_size)

        self.__initiate_all_clusters(cluster_2_grid)
        self.__update_model_clusters()

    def __initiate_all_clusters(self, cluster_2_grid: cluster_2_grid_dict) -> None:
        for index in cluster_2_grid.keys():
            self.clusters[index] = Cluster(index=index, members_grid=cluster_2_grid[index])

    def __update_model_clusters(self) -> None:
        cluster_keys_copy = list(self.clusters.keys())
        for index in cluster_keys_copy:
            cluster = self.clusters[index]
            eliminate = cluster.update(grid_2_data=self.grid_2_data,
                                       k_adjustments=self.k_adjustments,
                                       epsilon=self.epsilon)
            if eliminate:
                self.clusters.pop(index)


def get_search_grid_based_on_expansion(n_rows: int, n_columns: int,
                                       cluster_grid_mean: Tuple[float, float], cluster_memebrs,
                                       expansion_rate: float) -> List[Tuple[int, int]]:
    """This function returns a list of novel data near the cluster, the grid contains tuple coordinates
    of data points that are not already part of the cluster"""
    assert expansion_rate > 1
    proper_grid = {'row_lower': None,
                   'row_upper': None,
                   'col_lower': None,
                   'col_upper': None}
    row_dist_2_border = np.min(np.abs(np.asarray([cluster_grid_mean[0], cluster_grid_mean[0] - n_rows])))
    col_dist_2_border = np.min(np.abs(np.asarray([cluster_grid_mean[1], cluster_grid_mean[1] - n_columns])))

    row_expansion = int(row_dist_2_border * expansion_rate)
    col_expansion = int(col_dist_2_border * expansion_rate)

    proper_grid['row_lower'] = int(cluster_grid_mean[0] - row_expansion) \
        if cluster_grid_mean[0] - row_expansion > 0 else 0
    # upper bound is used in a list comprehension using the range() so n_rows-1 is not necessary
    proper_grid['row_upper'] = int(cluster_grid_mean[0] + row_expansion) \
        if cluster_grid_mean[0] + row_expansion < n_rows else n_rows

    proper_grid['col_lower'] = int(cluster_grid_mean[1] - col_expansion) \
        if cluster_grid_mean[1] - col_expansion > 0 else 0
    # upper bound is used in a list comprehension using the range() so n_columns-1 is not necessary
    proper_grid['col_upper'] = int(cluster_grid_mean[1] + col_expansion) \
        if cluster_grid_mean[1] + col_expansion < n_columns else n_columns

    grid_list = [(i, j)
                 for i in range(proper_grid['row_lower'], proper_grid['row_upper'])
                 for j in range(proper_grid['col_lower'], proper_grid['col_upper']) if (i, j) not in cluster_memebrs]
    return grid_list


def get_hostile_candidate_inclination(grid_list: List[Tuple[int, int]], grid_2_data: grid_2_data_dict, mean: NDArray,
                                      std: NDArray, k_adjustments: NDArray, gravity: float):
    """This function calculated exactly equation 11 from paper titled 'On the uphill battle of image frequency
    analysis' """
    try:
        data_minus_clustermean_squared = np.vstack([np.power(grid_2_data[grid] - mean, 2) for grid in grid_list])
    except ValueError:
        data_minus_clustermean_squared = np.reshape(np.asarray([np.power(grid_2_data[grid] - mean, 2)
                                                                for grid in grid_list]), (len(grid_list), 3))
    inv_std = np.power(std, -1)
    k_times_inv_std_squared = np.reshape((k_adjustments * inv_std) ** 2, (3, 1))

    inclination_arr = np.matmul(data_minus_clustermean_squared, k_times_inv_std_squared) ** 1.5
    inclination_arr = np.reshape(inclination_arr, (inclination_arr.shape[0]))
    inclination_arr = inclination_arr - gravity
    return inclination_arr


def get_cluster_candidate_inclination(data, mean, std, gravity, k_adjustments):
    arr = (data - mean) * k_adjustments * np.power(std, -1)
    arr = np.sum(np.power(arr, 2)) ** 1.5 - gravity
    return arr


TODO: "I should complete it like the paper i did. imagine wrapped around infinity"
