from typing import Dict, Tuple, List
from preprocessing import grid_tiling, load_image, get_grid_2_data
from numpy._typing import NDArray
from cluster import initiate_clustering, Cluster, update_all_clusters, GRAVITY_CONSTANT_3D
from collections import OrderedDict

# Maybe I should have a mapping class that contains all the dictionaries
#  all the refrences even k values or such, a container for all data they waY I use it


# you can do it based on snap shot of whole image or part of image
#  you can do it based on individual clusters. a for loop over all clusters. they
# all search in their neighborhood for points (or whole image because veins propagate through grid)
TODO: "Merge set--absorbtion dict- loss- dict" "to control the flow of code"

cluster_2_grid_dict = Dict[int, List[Tuple[int, int]]]
grid_2_data_dict = Dict[Tuple[int, int], NDArray]
grid_2_cluster_dict = Dict[Tuple[int, int], int]


class Model:
    clusters = OrderedDict()

    def __init__(self, image_url: str,
                 tile_size: int,
                 k_adjustments: NDArray,
                 epsilon: float = 10 ** -6,
                 scale_image: bool = True):
        self.image_url = image_url
        self.tile_size = tile_size
        self.k_adjustments = k_adjustments
        self.epsilon = epsilon
        self.scale_image = scale_image
        self.cluster_2_grid: Dict[int, List[Tuple[int, int]]] = {}
        self.grid_2_data: Dict[Tuple[int, int], NDArray] = {}
        self.grid_2_cluster: Dict[Tuple[int, int], int] = {}
        self.initiate_model()

    def initiate_model(self):
        loaded_image = load_image(self.image_url, self.scale_image, round_decimal=3)
        self.grid_2_data = get_grid_2_data(image=loaded_image)
        self.cluster_2_grid, self.grid_2_cluster = grid_tiling(grid_shape=loaded_image.shape, tile_size=self.tile_size)

        initiate_clustering(cluster_2_grid=self.cluster_2_grid, clusters=self.clusters)
        update_all_clusters(self.grid_2_data, self.k_adjustments, clusters=self.clusters)
