import math
from typing import List
from src.algo import FedSeq
import numpy as np
import random
from omegaconf import open_dict

def linear_growth(alpha, beta, round):
    return beta * (math.floor(alpha * (round - 1) + 1))
def log_growth(alpha, beta, round):
    return beta * (math.floor(alpha * math.log(round) + 1))
def exp_growth(alpha, beta, round):
    return beta * (math.floor((1 + alpha)**(round - 1)))
growth_functions = {'linear': linear_growth, 'log': log_growth, 'exp': exp_growth}

class FedSeqToParallel(FedSeq):  
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        self.alpha_growth = params.alpha_growth
        self.beta_growth = params.beta_growth
        self.growth_func = growth_functions[params.growth_func]
        with open_dict(params):
            params.clustering.n_clusters = math.floor(params.common.K / self.growth_func(self.alpha_growth, self.beta_growth, 1))
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
    
    def train_step(self):
        self.reassign_clients()
        self.writer.add_scalar("n_superclients", self.num_superclients, self._round)
        super().train_step()
    
    def reassign_clients(self):
        #NOTE TO SELF: for icg we need to sample the clients in order to create tot clusters of equal size and 
        n_new_superclients = self.growth_func(self.alpha_growth, self.beta_growth, self._round + 1)
        if  n_new_superclients > len(self.superclients):
            self.clustering_method._n_clusters = math.floor(self.num_clients / n_new_superclients)
            self.superclients = self.clustering_method.make_superclients(self.clients, self.representers, sub_path=self.evaluator.extract, **self.training,
                                        optimizer_class=self.optimizer, optimizer_args=self.optimizer_args)
            self.num_superclients = len(self.superclients)
        else:
            if self._round > 0:
                self._shuffle_current_superclients()
                
            
    def _shuffle_current_superclients(self):
        if self.clustering.classname != 'RandomClusterMaker':
            self._redistribute_clients_based_on_clusters()
        else:
            self.superclients = self.clustering_method.make_superclients(self.clients, self.representers, sub_path=self.evaluator.extract, **self.training,
                                        optimizer_class=self.optimizer, optimizer_args=self.optimizer_args)
            
    
    def _redistribute_clients_based_on_clusters(self):
        client_clusters_distribution = {i: [] for i in range(self.clustering_method._n_clusters)}
        client_superclients_distribution = {i: [] for i in range(len(self.superclients))}
        for n_superclient, superclient in enumerate(self.superclients):
            for index_client, c in enumerate(superclient.clients):
                client_clusters_distribution[c.cluster_id].append((n_superclient, index_client))
                client_superclients_distribution[n_superclient].append(c.cluster_id)
        for n_superclient, cluster_labels in client_superclients_distribution.items():
            for i, c_label in enumerate(cluster_labels):
                rand_index = np.random.choice(range(len(client_clusters_distribution[c_label])))
                (wanted_sc, wanted_c) = client_clusters_distribution[c_label].pop(rand_index)
                self.superclients[n_superclient].clients[i] = self.superclients[wanted_sc].clients[wanted_c]
            random.shuffle(self.superclients[n_superclient].clients)
