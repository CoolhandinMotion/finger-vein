from collections import defaultdict
from dataclasses import dataclass
import time
import numpy as np
from typing import Dict, Protocol, Set, Tuple
from numpy.typing import NDArray
from model import Model, get_search_grid_based_on_expansion,\
    get_hostile_candidate_inclination, get_cluster_candidate_inclination, grid_2_data_dict

@dataclass
class Cluster(Protocol):
    mean: float
    members_grid: Set[Tuple[int, int]]

    def update(self, grid_2_data: grid_2_data_dict,
               k_adjustments: NDArray, epsilon: float) -> bool:
        ...


class Control:
    def __init__(self, model: Model, n_epochs: int = 50, merge_threshold: float = 1, expansion_rate: float = 2):
        self.model = model
        self.n_epoch = n_epochs
        self.merge_threshold = merge_threshold
        self.expansion_rate = expansion_rate

    def handle_hostile_merges(self, hostile_index: int, merge_set: set):
        hostile_cluster = self.model.clusters[hostile_index]
        # eliminated_cluster_indices = set()
        for host_index in merge_set:
            host_cluster = self.model.clusters[host_index]
            if np.abs(hostile_cluster.mean[-1] - host_cluster.mean[-1]) < self.merge_threshold:
                # change the associated cluster for exchanged data from host to hostile
                for grid in host_cluster.members_grid:
                    self.model.grid_2_cluster[grid] = hostile_index
                #  add the new members to the hostile cluster members
                hostile_cluster.members_grid.update(host_cluster.members_grid)
                # remove the cluster that was merged from consideration
                self.model.clusters.pop(host_index)

        hostile_cluster.update(self.model.grid_2_data,self.model.k_adjustments,self.model.epsilon)

    def handle_hostile_absorptions(self, hostile_index: int, absorption_set: set):
        hostile_cluster = self.model.clusters[hostile_index]
        for grid in absorption_set:
            self.model.grid_2_cluster[grid] = hostile_index
        hostile_cluster.members_grid.update(absorption_set)
        hostile_cluster.update(self.model.grid_2_data,self.model.k_adjustments,self.model.epsilon)

    def handle_host_losses(self, loss_dict:Dict[int, set]):
        for host_index in loss_dict:
            host_cluster = self.model.clusters[host_index]
            host_cluster.members_grid.difference(loss_dict[host_index])
            host_cluster.update(self.model.grid_2_data,self.model.k_adjustments,self.model.epsilon)

    def run(self):
        for epoch in range(self.n_epoch):
            epoch_start = time.time()
            present_clusters = list(self.model.clusters.keys())
            print(f"Epoch ... {epoch} ... number of clusters ... {len(present_clusters)}... ")
            for hostile_index in present_clusters:
                if hostile_index not in self.model.clusters:
                    continue

                absorption_set: Set[Tuple[int, int]] = set()
                merge_set : Set[int] = set()
                loss_dict: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

                hostile_cluster = self.model.clusters[hostile_index]
                TODO: "code below is good but too slow"
                grid_list = get_search_grid_based_on_expansion(n_rows=self.model.n_rows,
                                                               n_columns=self.model.n_columns,
                                                               cluster_grid_mean=hostile_cluster.mean[:2],
                                                               cluster_memebrs=hostile_cluster.members_grid,
                                                               expansion_rate=self.expansion_rate)
                # grid_list = list(self.model.grid_2_data.keys())
                data_inclination_hostile = get_hostile_candidate_inclination(grid_list=grid_list,
                                                                             grid_2_data=self.model.grid_2_data,
                                                                             mean=hostile_cluster.mean,
                                                                             std=hostile_cluster.std,
                                                                             k_adjustments=self.model.k_adjustments,
                                                                             gravity=hostile_cluster.gravity)
                for i, value in enumerate(data_inclination_hostile):
                    if value < 0:
                        grid = grid_list[i]
                        data = self.model.grid_2_data[grid]
                        host_index = self.model.grid_2_cluster[grid]
                        try:
                            host = self.model.clusters[host_index]
                        except KeyError:
                            absorption_set.add(grid)
                            continue
                        host_inclination = get_cluster_candidate_inclination(data=data,
                                                                             mean=host.mean,
                                                                             std=host.std,
                                                                             gravity=host.gravity,
                                                                             k_adjustments=self.model.k_adjustments)
                        if host_inclination < 0:
                            merge_set.add(host_index)

                        else:
                            absorption_set.add(grid)
                            loss_dict[host_index].add(grid)

                self.handle_hostile_absorptions(hostile_index=hostile_index, absorption_set=absorption_set)
                self.handle_host_losses(loss_dict=loss_dict)
                self.handle_hostile_merges(hostile_index=hostile_index, merge_set=merge_set)
            print(f"it took..{np.around(time.time() - epoch_start, decimals=2)} Seconds")
            print("/"*36)

