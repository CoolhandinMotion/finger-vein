import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import cv2
import time
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from statsmodels.stats import stattools
import gc

def initial_tiling(img):
    # /////dic[cluster] holds a set of flat_indexed indices of pixels which belong to the cluster///////
    arr = img
    c = 1
    step_size = img.shape[1]
    for i in range(arr.shape[0] // tile_size):
        for j in range(arr.shape[1] // tile_size):
            # dic[c]=arr[tile_size*i:tile_size*(i+1),tile_size*j:tile_size*(j+1)].flatten()
            dic[c] = set([t * step_size + k for t in range(tile_size * i, tile_size * (i + 1)) for k in
                          range(tile_size * j, tile_size * (j + 1))])
            initiate_reverse_dic((tile_size * i, tile_size * (i + 1)), (tile_size * j, tile_size * (j + 1)), c)
            c += 1

    # //////handling extra misfit rows///////
    col_misfit_index = (arr.shape[1] // tile_size) * tile_size
    row_misfit_index = (arr.shape[0] // tile_size) * tile_size
    if arr.shape[0] % tile_size != 0:
        col_step = (tile_size ** 2) // (arr.shape[0] - row_misfit_index)
        if col_step < col_misfit_index:
            for i in range(col_misfit_index // col_step):
                if not i == col_misfit_index // col_step - 1:
                    # dic[c]=arr[row_misfit_index:,col_step*i:col_step*(i+1)].flatten()
                    dic[c] = set([t * step_size + k for t in range(row_misfit_index, arr.shape[0]) for k in
                                  range(col_step * i, col_step * (i + 1))])
                    initiate_reverse_dic((row_misfit_index, arr.shape[0]), (col_step * i, col_step * (i + 1)), c)
                    c += 1

                else:
                    # dic[c] = arr[row_misfit_index:,col_step*i:col_misfit_index].flatten()
                    dic[c] = set([t * step_size + k for t in range(row_misfit_index, arr.shape[0]) for k in
                                  range(col_step * i, col_misfit_index)])
                    initiate_reverse_dic((row_misfit_index, arr.shape[0]), (col_step * i, col_misfit_index), c)
                    c += 1
        else:

            # dic[c]=arr[row_misfit_index:,:col_misfit_index].flatten()
            dic[c] = set(
                [t * step_size + k for t in range(row_misfit_index, arr.shape[0]) for k in range(col_misfit_index)])
            initiate_reverse_dic((row_misfit_index, arr.shape[0]), (0, arr.shape[1]), c)
            c += 1

    if arr.shape[1] % tile_size != 0:
        row_step = (tile_size ** 2) // (arr.shape[1] - col_misfit_index)
        if row_step < row_misfit_index:
            for i in range(row_misfit_index // row_step):
                if not i == row_misfit_index // row_step - 1:
                    # dic[c]=arr[row_step*i:row_step*(i+1),col_misfit_index:].flatten()
                    dic[c] = set([t * step_size + k for t in range(row_step * i, row_step * (i + 1)) for k in
                                  range(col_misfit_index, arr.shape[1])])
                    initiate_reverse_dic((row_step * i, row_step * (i + 1)), (col_misfit_index, arr.shape[1]), c)
                    c += 1
                else:
                    # dic[c] = arr[row_step*i:row_misfit_index,col_misfit_index:].flatten()
                    dic[c] = set([t * step_size + k for t in range(row_step * i, row_misfit_index) for k in
                                  range(col_misfit_index, arr.shape[1])])
                    initiate_reverse_dic((row_step * i, row_misfit_index), (col_misfit_index, arr.shape[1]), c)
                    c += 1

        else:

            # dic[c]=arr[:row_misfit_index,col_misfit_index:].flatten()
            dic[c] = set(
                [t * step_size + k for t in range(row_misfit_index) for k in range(col_misfit_index, arr.shape[1])])
            initiate_reverse_dic((0, row_misfit_index), (col_misfit_index, arr.shape[1]), c)
            c += 1

    if arr.shape[0] % tile_size != 0 and arr.shape[1] % tile_size != 0:
        # dic[c]=arr[row_misfit_index:,col_misfit_index:].flatten()
        dic[c] = set([t * step_size + k for t in range(row_misfit_index, arr.shape[0]) for k in
                      range(col_misfit_index, arr.shape[1])])
        initiate_reverse_dic((row_misfit_index, arr.shape[0]), (col_misfit_index, arr.shape[1]), c)


def initiate_reverse_dic(row_range, column_range, cluster):
    # ////Here we consider flattened image indexes. Array by the shape (img.shape[0]*img.shape[1],)///////
    step_size = img.shape[1]
    for i in range(row_range[0], row_range[1]):
        for j in range(column_range[0], column_range[1]):
            reverse_dic[i * step_size + j] = cluster


def update_cluster_ref_dict(c, feedback, k, enable_entropy=True):
    if c not in dic:
        return
    cluster_value_array = np.array([flatten_img[i] for i in dic[c]])
    mean = np.mean(cluster_value_array)
    std = np.std(cluster_value_array)
    if feedback:
        t_square = (1 / (1 + k * std)) ** 2
    else:
        t_square = t ** 2
    gravity = gravity_constant * (std ** 3) * t_square * np.exp(t_square / 2)
    if enable_entropy:
        if not std == 0:
            entropy = .5 * np.log(2 * np.pi * np.e * std ** 2)
        else:
            entropy = 0
        ref_dic[c] = [mean, std, gravity, entropy]
    else:
        ref_dic[c] = [mean, std, gravity]


def inclination_snap_shot():
    shuffled_cluster_key_list = random.sample(list(ref_dic.keys()), len(ref_dic.keys()))
    # ////constructing inclination matrix with dimension of (pixel_num,cluster_num)//////
    pixel_matrix = np.transpose(np.array([flatten_img for j in range(len(ref_dic.keys()))]))
    cluster_mean_array = np.array([ref_dic[key][0] for key in shuffled_cluster_key_list])
    cluster_mean_matrix = np.array([cluster_mean_array for j in range(len(flatten_img))])
    inclination_matrix = np.abs(pixel_matrix - cluster_mean_matrix) ** 3
    del cluster_mean_array
    del cluster_mean_matrix
    del pixel_matrix
    gc.collect()
    # ////constructing gravity matrix with dimension of (pixel_num,cluster_num)//////
    gravity_array = np.array([ref_dic[key][2] for key in shuffled_cluster_key_list])
    gravity_matrix = np.array([gravity_array for j in range(len(flatten_img))])

    return inclination_matrix - gravity_matrix, shuffled_cluster_key_list


def ISMS_1d(epoch=10):
    for i in range(epoch):
        exchanged_pixel_set.clear()
        only_merged_pixel_set.clear()
        epoch_start = time.time()
        entropy_sum = np.around(sum([ref_dic[x][3] for x in ref_dic]), decimals=2)
        total_entropy_during_learning.append(entropy_sum)

        snap_matrix, shuffled_cluster_key = inclination_snap_shot()
        merge_set, absorption_dict, loss_dict = processing_absorption_merge(snap_matrix, shuffled_cluster_key)
        handle_absorptions(absorption_dict, loss_dict)
        handle_merges(merge_set)
        del snap_matrix
        del merge_set
        del absorption_dict
        del loss_dict
        total_pixels_exchanged_during_epoch.append(len(exchanged_pixel_set))
        total_merges_happened_during_epoch.append(len(only_merged_pixel_set))
        entropy_sum = sum([ref_dic[x][3] for x in ref_dic])
        print(f"Ending Entropy was..{np.around(entropy_sum, decimals=3)}..in total")
        print(f"This was epoch...{i}...and it took..{np.around(time.time() - epoch_start, decimals=2)} Seconds")
        print("///////////////////////////////////////")
        if len(exchanged_pixel_set) < total_pixel_num * portion_pixel_exchanged_to_halt:
            print(f"\nNot enough pixel exchanges. Exiting ISMS loop\n")
            break


def handle_absorptions(absorption_dict, loss_dict):
    # ///absorption_dict contains flattened_pixel_indices for each cluster to be added////
    for cluster in list(absorption_dict.keys()):
        dic[cluster].update(absorption_dict[cluster])
        exchanged_pixel_set.update(absorption_dict[cluster])
        update_cluster_ref_dict(cluster, feedback, k)
        for pixel_index in absorption_dict[cluster]:
            reverse_dic[pixel_index] = cluster
    # ///loss_dict contains flattened_pixel_indices for each cluster to be removed////
    for cluster in list(loss_dict.keys()):
        dic[cluster].difference_update(loss_dict[cluster])
        if len(dic[cluster]) == 0:
            del dic[cluster]
            del ref_dic[cluster]
        update_cluster_ref_dict(cluster, feedback, k)

    # for cluster in set(absorption_dict.keys()).union(set(loss_dict.keys())):
    #     update_cluster_ref_dict(cluster,feedback,k)


def handle_merges(merge_set):
    for frozen_set in merge_set:
        frz = list(frozen_set)
        c1, c2 = frz[0], frz[1]
        if (c1 in ref_dic) and (c2 in ref_dic) and (c1 != c2):
            if np.abs(ref_dic[c1][0] - ref_dic[c2][0]) < merge_threshold:
                dic[c1].update(dic[c2])
                exchanged_pixel_set.update(dic[c2])
                only_merged_pixel_set.update(dic[c2])
                for pixel_index in dic[c2]:
                    reverse_dic[pixel_index] = c1
                update_cluster_ref_dict(c1, feedback, k)
                del ref_dic[c2]
                del dic[c2]


def processing_absorption_merge(snap_matrix, shuffled_cluster_key):
    loss_dict = defaultdict(set)
    absorption_dict = defaultdict(set)
    merge_set = set()
    # //////////////////////////////////////////////////
    # cluster_name_list=list(ref_dic.keys())
    if len(shuffled_cluster_key) < 3:
        print("Algorithm collapsed into a Black Hole!!!")
        write_txt_report()
        quit()
    min_cluster_index = np.argmin(snap_matrix, axis=1)
    for flatten_index in range(total_pixel_num):
        host_cluster_name = reverse_dic[flatten_index]
        if host_cluster_name not in ref_dic:
            continue
        candid_cluster_name = shuffled_cluster_key[min_cluster_index[flatten_index]]
        if candid_cluster_name == host_cluster_name:
            continue
        inclination_toward_host = snap_matrix[flatten_index][shuffled_cluster_key.index(host_cluster_name)]
        inclination_toward_candid = snap_matrix[flatten_index][min_cluster_index[flatten_index]]
        if inclination_toward_host >= 0 and inclination_toward_candid <= 0:
            if len(dic[host_cluster_name]) <= tile_size or len(dic[candid_cluster_name]) <= tile_size:
                merge_set.add(frozenset([host_cluster_name, candid_cluster_name]))
                continue
            else:
                absorption_dict[candid_cluster_name].add(flatten_index)
                loss_dict[host_cluster_name].add(flatten_index)
                continue
        elif inclination_toward_host < 0 and inclination_toward_candid < 0:
            merge_set.add(frozenset([host_cluster_name, candid_cluster_name]))
    del shuffled_cluster_key
    del min_cluster_index
    del snap_matrix

    return merge_set, absorption_dict, loss_dict


def write_txt_report(shapiro_score=False):
    np.set_printoptions(suppress=True)
    number_pixel_in_dic = 0
    for cluster in dic:
        number_pixel_in_dic += len(dic[cluster])
    file = open(str(output_folder_url + "\Report.txt"), "w")
    file.write(f"Total number of pixels were...{total_pixel_num}\n")
    file.write(f"Total number of clusters is..{len(ref_dic.keys())}\n\n\n")
    for cluster in ref_dic:
        file.write(
            f"cluster...{cluster}  has {len(dic[cluster])} members.\n "
            f"{np.around(ref_dic[cluster][:2], decimals=3)}\n")
        if shapiro_score:
            all_data_in_cluster=np.array([flatten_img[i] for i in dic[cluster]])
            shapiro_score=list(stattools.stats.shapiro(all_data_in_cluster))[1]
            file.write(f"Shapiro score is...{shapiro_score:.2f}...")
            if shapiro_score <.05:
                file.write(f"NOT Gaussian\n")
            else:
                file.write(f"IS indeed Gaussian\n")
        file.write("-----------------------------------------------------------------\n")

def plot_learning_dynamics(plot_hist=False):

    ax=plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x = np.arange(0, len(total_entropy_during_learning))
    # ///////////////////////////////////////////////////
    y = np.array(total_entropy_during_learning)
    plt.plot(x, y,color="g")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.savefig(str(output_folder_url + "\Total_Entropy_epoch.png"))
    plt.close()
#     ///////////////////////////////////////
    y=100*(np.array(total_merges_happened_during_epoch)/np.array(total_pixels_exchanged_during_epoch))
    plt.plot(x,y,color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Merge % all changes")
    plt.savefig(str(output_folder_url + "\Merge_percent_epoch.png"))
    plt.close()

    if plot_hist:
        for cluster in ref_dic:
            arr=np.array([flatten_img[i] for i in dic[cluster]])
            plt.hist(arr,[i for i in range(256)],color="b")
            plt.savefig(str(output_folder_url + f"\Cluster_Histogram\{cluster}.png"))
            plt.close()


algorithm_start = time.time()
dic = defaultdict(set)
ref_dic = defaultdict()
reverse_dic = defaultdict()
exchanged_pixel_set = set()
only_merged_pixel_set=set()
# ////parameters////
feedback = False
k = 100
gravity_constant = np.sqrt(2 * np.pi)

t = 1
tile_size = 100
merge_threshold = 1

portion_pixel_exchanged_to_halt = .01
img_url = r"E:\PycharmProjects\Dr-Kauba-code\02_002_R-DORSAL_02_TI.png"
output_folder_url = r"E:\PycharmProjects\Dr-Kauba-code\bearbeited"
# ////////////////////////////////////////
img = cv2.imread(img_url, 0)
print(f"Image dimension is..{img.shape}")
flatten_img = img.flatten()
total_pixel_num = img.shape[0] * img.shape[1]
# ////////////////////////
initial_tiling(img)

for cluster in dic:
    update_cluster_ref_dict(cluster, feedback, k)

# ///////Running the Algorithm////////
total_entropy_during_learning = []
total_pixels_exchanged_during_epoch = []
total_merges_happened_during_epoch = []
ISMS_1d(epoch=50)
plot_learning_dynamics(plot_hist=True)
write_txt_report()
print(f"Whole algorithm took...{np.around(time.time() - algorithm_start, decimals=2)} Seconds")