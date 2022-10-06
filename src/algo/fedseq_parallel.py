import math
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
        self.avg_n_superclients = 0
        with open_dict(params):
            params.clustering.n_clusters = math.floor(params.common.K / self.growth_func(self.alpha_growth, self.beta_growth, 1))
            params.clustering.n_superclients = self.growth_func(self.alpha_growth, self.beta_growth, 1)
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        if self.clustering.collect_time_statistics:
            self.avg_largest_sc_n_examples = 0
            self.avg_largest_sc_n_clients = 0
    
    def train_step(self):
        self.reassign_clients()
        self.avg_n_superclients += self.num_superclients
        self.writer.add_scalar("n_superclients", self.num_superclients, self._round + 1)
        if self.clustering.collect_time_statistics:
            self.avg_largest_sc_n_examples += self.writer.local_store['largest_sc_#_examples']
            self.avg_largest_sc_n_clients += self.writer.local_store['largest_sc_#_clients']
        super().train_step()
    
    def reassign_clients(self):
        n_new_superclients = self._check_validity(self.growth_func(self.alpha_growth, self.beta_growth, self._round + 1))
        if  n_new_superclients > len(self.superclients):
            self.clustering_method._n_clusters = math.floor(self.num_clients / n_new_superclients)
            self.clustering_method._n_superclients = n_new_superclients
            self.superclients = self.clustering_method.make_superclients(self.clients, self.representers, sub_path=self.evaluator.extract, **self.training,
                                        optimizer_class=self.optimizer, optimizer_args=self.optimizer_args, one_time_clustering = (not self.keep_representers))
            self.num_superclients = len(self.superclients)
    
    def _check_validity(self, n_superclients):
        if n_superclients <= self.num_clients:
            return n_superclients
        return self.num_clients