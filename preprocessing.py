from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List
from PIL import Image



def grayscale_tile(image: NDArray, tile_size: int) -> Dict[int, List[NDArray]]:
    cluster_2_data_dict = defaultdict()
    assert len(image.shape) == 2
    assert isinstance(tile_size,int)
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    first_iter_over_rows = int(n_rows // tile_size)
    first_iter_over_cols = int(n_cols // tile_size)

    cluster_index = 1
    for i in range(first_iter_over_rows):
        for j in range(first_iter_over_cols):
            cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]]) for t in
                                                  range(i * tile_size, (i + 1) * tile_size)
                                                  for k in
                                                  range(j * tile_size, (j + 1) * tile_size)]
            cluster_index += 1

    # if 2d Data shape is totally divisible by tile_size we are done here
    if n_rows % tile_size == 0 and n_cols % tile_size == 0:
        return cluster_2_data_dict

    # if the 2d Data shape divided by tile_size produces some leftover we need to cover that as well

    first_row_index_not_covered = first_iter_over_rows * tile_size
    first_col_index_not_covered = first_iter_over_cols * tile_size

    if n_rows % tile_size != 0:
        ideal_col_length = (tile_size ** 2) // (n_rows - first_row_index_not_covered)
        if ideal_col_length < first_col_index_not_covered:
            last_iter_over_cols = first_col_index_not_covered // ideal_col_length
            for i in range(last_iter_over_cols):
                cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                                      for t in
                                                      range(first_row_index_not_covered, n_rows)
                                                      for k in range(ideal_col_length * i,
                                                                     ideal_col_length * (i + 1))]
                cluster_index += 1

            cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                                  for t in range(first_row_index_not_covered, n_rows)
                                                  for k in
                                                  range(last_iter_over_cols * ideal_col_length,
                                                        first_col_index_not_covered)]
            cluster_index += 1
        else:
            cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                                  for t in range(first_row_index_not_covered, n_rows)
                                                  for k in range(first_col_index_not_covered)]
            cluster_index += 1

    if n_cols % tile_size != 0:
        ideal_row_length = (tile_size ** 2) // (n_cols - first_col_index_not_covered)

        if ideal_row_length < first_row_index_not_covered:
            last_iter_over_rows = first_row_index_not_covered // ideal_row_length
            for i in range(last_iter_over_rows):
                cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                                      for t in range(ideal_row_length * i,
                                                                     ideal_row_length * (i + 1))
                                                      for k in
                                                      range(first_col_index_not_covered, n_cols)]
                cluster_index += 1

            cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                                  for t in
                                                  range(ideal_row_length * last_iter_over_rows,
                                                        first_row_index_not_covered)
                                                  for k in range(first_col_index_not_covered, n_cols)]
            cluster_index += 1

        else:
            cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                                  for t in range(first_row_index_not_covered)
                                                  for k in range(first_col_index_not_covered, n_cols)]
            cluster_index += 1

    if n_rows % tile_size != 0 and n_cols % tile_size != 0:
        cluster_2_data_dict[cluster_index] = [np.asarray([t, k, image[t][k]])
                                              for t in range(first_row_index_not_covered, n_rows)
                                              for k in range(first_col_index_not_covered, n_cols)]
    return cluster_2_data_dict


def load_image(image_url: str, standard_scaling = False, round_decimal: int = 6) -> NDArray:
    image = Image.open(image_url)
    image_array = np.asarray(image)
    if standard_scaling:
        image_array = np.around(image_array / 255,decimals=round_decimal)
    return image_array

def get_image_2_data_dict(image: NDArray) -> dict:
    image_2_data_dict = {(i,j):np.asarray([i,j,image[i][j]])
                         for i in range(image.shape[0])
                         for j in range(image.shape[1])}
    return image_2_data_dict
