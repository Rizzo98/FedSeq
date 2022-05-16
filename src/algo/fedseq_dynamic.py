from src.algo import FedSeq
import numpy as np
from collections import defaultdict
import copy
from random import shuffle
from math import floor, ceil
class FedSeqDynamic(FedSeq):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        self.clusters = defaultdict(set)
        self.p_shuffling = (self.num_superclients-1)/self.num_superclients
        for s in self.superclients:
            for i, c in enumerate(s.clients):
                self.clusters[c.cluster_id].add((s.client_id, i))

    def train_step(self):
        '''
        Reminder: we save (superclient, index of client) in the dictionary of the nth cluster. So when we 
        do the substitution, we just need to swap them in the .clients and at those positions in the 2
        superclients still there are clients of the n-th cluster! CHECK'''
        for cluster_id in self.clusters.keys():
            clients_in_cluster = list(self.clusters[cluster_id])
            n_ids = len(clients_in_cluster) 
            shuffle(clients_in_cluster)
            first_half = clients_in_cluster[:floor(n_ids/2)]
            second_half = clients_in_cluster[ceil(n_ids/2):]
            for i in range(len(first_half)):
                if np.random.uniform() > 1 - self.p_shuffling:
                    c = self.superclients[first_half[i][0]].clients[first_half[i][1]]
                    self.superclients[first_half[i][0]].clients[first_half[i][1]] = self.superclients[second_half[i][0]].clients[second_half[i][1]]
                    self.superclients[second_half[i][0]].clients[second_half[i][1]] = c
        super().train_step()
        