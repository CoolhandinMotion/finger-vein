from numpy.typing import NDArray, ArrayLike
from typing import List, Tuple, Dict, Set
import numpy as np

GRAVITY_CONSTANT_3D = np.sqrt(2 * np.pi ** 3)

cluster_2_gird_dict = Dict[int, Set[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
grid_2_cluster_dict = Dict[Tuple[int, int], int]


class Cluster:
    # data is all three dimensional data that is held my members
    # mambers_grid is just (i,j) tuple thats serves as key and coordinates of pixels
    def __init__(self, index: int, members_grid: Set[Tuple[int, int]]):
        self.index = index
        self.members_grid = members_grid
        self.data = None
        self.mean = None
        self.gravity = None
        self.inv_cov_matrix = None
        self.std = None

    def update(self, grid_2_data: grid_2_data_dict,
               k_adjustments: NDArray, epsilon: float) -> bool:

        if len(self.members_grid) <= 1:
            return True

        self.__update_data(grid_2_data=grid_2_data)
        self.__update_mean()
        self.__update_std(epsilon=epsilon)
        self.__update_gravity(k_adjustments=k_adjustments)
        return False

    def __update_data(self, grid_2_data: grid_2_data_dict) -> None:
        try:
            cluster_arr = np.vstack([grid_2_data[grid] for grid in self.members_grid])
            self.data = cluster_arr
        except ValueError:
            ...

    def __update_mean(self) -> None:
        self.mean = np.mean(self.data, axis=0)

    def __update_std(self, epsilon: float) -> None:
        std = np.std(self.data, axis=0)
        if any(std) == 0:
            self.std = np.asarray(list(map(lambda x: epsilon if x == 0 else x, std)))
        else:
            self.std = std

    def __update_gravity(self, k_adjustments: NDArray) -> None:
        k_times_sigma = self.std * k_adjustments
        adjusted_force_array = (np.power(k_times_sigma + 1, -1) * k_adjustments) ** 2
        force = np.sum(adjusted_force_array)
        det = np.product(k_times_sigma)
        self.gravity = GRAVITY_CONSTANT_3D * det * force * np.exp(force / 2)
