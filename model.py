from typing import Dict, Tuple, List, Set
from preprocessing import grid_tiling, load_image, get_grid_2_data
from numpy._typing import NDArray
from cluster import initiate_clustering, Cluster, update_all_clusters
from collections import OrderedDict
import numpy as np

# Maybe I should have a mapping class that contains all the dictionaries
#  all the refrences even k values or such, a container for all data they waY I use it


# you can do it based on snap shot of whole image or part of image
#  you can do it based on individual clusters. a for loop over all clusters. they
# all search in their neighborhood for points (or whole image because veins propagate through grid)
TODO: "Merge set--absorbtion dict- loss- dict" "to control the flow of code"

cluster_2_grid_dict = Dict[int, Set[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
grid_2_cluster_dict = Dict[Tuple[int, int], int]

GRAVITY_CONSTANT_3D = np.sqrt(2 * np.pi ** 3)


class Model:
    clusters = OrderedDict()

    def __init__(self, image_url: str,
                 tile_size: int,
                 k_adjustments: NDArray,
                 epsilon: float = 10 ** -6,
                 scale_image: bool = True):
        self.n_columns = None
        self.n_rows = None
        self.image_url = image_url
        self.tile_size = tile_size
        self.k_adjustments = k_adjustments
        self.epsilon = epsilon
        self.scale_image = scale_image
        self.cluster_2_grid: Dict[int, Set[Tuple[int, int]]] = {}
        self.grid_2_data: Dict[Tuple[int, int], NDArray] = {}
        self.grid_2_cluster: Dict[Tuple[int, int], int] = {}
        self.initiate_model()

    def initiate_model(self) -> None:
        loaded_image = load_image(self.image_url, self.scale_image, round_decimal=3)
        self.n_rows, self.n_columns = loaded_image.shape[0], loaded_image.shape[1]
        self.grid_2_data = get_grid_2_data(image=loaded_image)
        self.cluster_2_grid, self.grid_2_cluster = grid_tiling(grid_shape=loaded_image.shape, tile_size=self.tile_size)

        initiate_clustering(cluster_2_grid=self.cluster_2_grid, clusters=self.clusters)
        update_all_clusters(self.grid_2_data,
                            self.k_adjustments,
                            clusters=self.clusters,
                            epsilon=self.epsilon)


def get_grid_limits_for_cluster(n_rows: int, n_columns: int,
                           cluster_grid_mean: Tuple[float, float],
                           expansion_rate: float) -> Dict[str, int]:
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

    proper_grid['row_upper'] = int(cluster_grid_mean[0] + row_expansion) \
        if cluster_grid_mean[0] + row_expansion < n_rows - 1 else n_rows - 1

    proper_grid['col_lower'] = int(cluster_grid_mean[1] - col_expansion) \
        if cluster_grid_mean[1] - col_expansion > 0 else 0

    proper_grid['col_upper'] = int(cluster_grid_mean[1] + col_expansion) \
        if cluster_grid_mean[1] + col_expansion < n_columns - 1 else n_columns - 1

    return proper_grid

def get_candidate_data_for_cluster(grid_limits_for_cluster,grid_2_data,):
    ...



TODO: "I should complete it like the paper i did. imagine wrapped around infinity"


def get_wrapped_grid_for_data_candidates(n_rows: int, n_columns: int,
                                         cluster_grid_mean: Tuple[float, float],
                                         expansion_rate: float) -> Dict[str, int]:
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

    proper_grid['row_upper'] = int(cluster_grid_mean[0] + row_expansion) \
        if cluster_grid_mean[0] + row_expansion < n_rows - 1 else n_rows - 1

    proper_grid['col_lower'] = int(cluster_grid_mean[1] - col_expansion) \
        if cluster_grid_mean[1] - col_expansion > 0 else 0

    proper_grid['col_upper'] = int(cluster_grid_mean[1] + col_expansion) \
        if cluster_grid_mean[1] + col_expansion < n_columns - 1 else n_columns - 1

    return proper_grid
