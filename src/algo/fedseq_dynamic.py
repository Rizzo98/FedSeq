from src.algo import FedSeq
import numpy as np
from collections import defaultdict
import copy
from random import shuffle
from math import floor, ceil
class FedSeqDynamic(FedSeq):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        self.clusters = defaultdict(list)
        self.p_shuffling = 0.5
        for s in self.superclients:
            for c in s.clients:
                self.clusters[c.cluster_id].append((s.client_id, c.client_id))

    def train_step(self):
        for cluster_id in self.clusters.keys():
            n_ids = len(self.clusters[cluster_id]) 
            shuffle(self.clusters[cluster_id])
            first_half = self.clusters[cluster_id][:floor(n_ids/2)]
            second_half = self.clusters[cluster_id][ceil(n_ids/2):]
            for i in range(len(first_half)):
                if np.random.uniform() > 1 - self.p_shuffling:
                    c = self.superclients[first_half[i][0]].clients[first_half[i][1]]
                    self.superclients[first_half[i][0]].clients[first_half[i][1]] = self.superclients[second_half[i][0]].clients[second_half[i][1]]
                    self.superclients[second_half[i][0]].clients[second_half[i][1]] = c
        super().train_step()
        