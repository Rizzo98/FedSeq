import datetime
import os
import time
from typing import List, Tuple, Any, Callable
from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from sklearn.decomposition import PCA
from src.algo.fed_clients import Client
from scipy.spatial.distance import euclidean
import math
import itertools as it
from abc import ABC, abstractmethod
from src.algo.fedseq_modules import FedSeqSuperClient, GraphSuperclient
import logging
from src.datasets.cifar import CifarLocalDataset
from src.datasets.shakespeare import ShakespeareLocalDataset
from src.utils import savepickle
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
log = logging.getLogger(__name__)


class ClientCluster:
    def __init__(self, id, representer_len=None, logger=log):
        self.clients = []
        self.clients_representer = []
        self.id = id
        self.__num_examples = 0
        self.representer_len = representer_len
        self.log = logger

    def add_client(self, client, representer: np.ndarray or None = None):
        self.clients.append(client)
        self.clients_representer.append(representer)
        self.__num_examples += len(client)

    def confidence(self) -> np.ndarray:
        assert self.representer_len is not None, "Vector dimension needed to calculate cluster vector"
        if len(self.clients) == 0:
            return np.full(self.representer_len, 1 / self.representer_len)
        cluster_representer = np.zeros(self.representer_len)
        clients_dataset_len = np.sum([len(c) for c in self.clients])
        for c, p in zip(self.clients, self.clients_representer):
            cluster_representer = cluster_representer + np.array(p) * len(c) / clients_dataset_len
        return cluster_representer

    def make_superclient(self, verbose: bool = False, **sup_kwargs):
        if verbose:
            self.log.info(f"superclient {self.id}: examples: {self.__num_examples}, clients: {self.num_clients()}")
        return FedSeqSuperClient(self.id, self.clients, **sup_kwargs)

    def num_clients(self) -> int:
        return len(self.clients)

    def num_examples(self) -> int:
        return self.__num_examples

    def pop(self, index: int = -1) -> Tuple[Client, List[int]]:
        client = self.clients.pop(index)
        client_representer = self.clients_representer.pop(index)
        self.__num_examples -= len(client)
        return client, client_representer

    def clients_id(self) -> List[int]:
        return [c.client_id for c in self.clients]


class ClusterMaker(ABC):
    def __init__(self, min_examples: int, max_clients: int, save_statistics: bool, savedir: str,
                    measure: str = None, verbose: bool = False, collect_time_statistics: bool = False, save_visualization: bool = False, writer = None, *args, **kwargs):
        self._min_examples = min_examples
        self._max_clients = max_clients
        self._save_statistics = save_statistics
        self._savedir = savedir
        self._measure = measure
        self._writer = writer
        self.verbose = verbose
        self._save_visualization = save_visualization
        self._collect_time_statistics = collect_time_statistics
        self._statistics = {}

    @property
    def measure(self):
        return self._measure

    @measure.setter
    def measure(self, measure):
        self._measure = measure

    @property
    def save_statistic(self):
        return self._save_statistics

    @save_statistic.setter
    def save_statistic(self, save_statistics):
        self._save_statistics = save_statistics

    def make_superclients(self, clients: List[Client], representers: List[np.ndarray], sub_path: str = "", one_time_clustering: bool = False, **sup_kwargs) \
            -> List[FedSeqSuperClient]:
        self._statistics.clear()
        num_classes = clients[0].num_classes
        assert all(num_classes == c.num_classes for c in clients), "Clients have different label space's dimension"
        start = time.time()
        clusters = self._make_clusters(clients, representers)
        end = time.time()
        sp = [c.make_superclient(self.verbose, num_classes=num_classes, **sup_kwargs) for c in clusters]
        self._save_time_statistics(one_time_clustering, end-start, sp)
        if self._save_statistics:
            self._collect_clustering_statistics(clients, ("superclients", {i: s.num_ex_per_class()
                                                                       for i, s in enumerate(sp)}))
        self._save_tsne(clients, representers, sp)
        savepickle(self._statistics,
                   os.path.join(self._savedir, sub_path, f"{self.__class__.__name__}_{self._measure}_stats.pkl"))
        return sp

    @abstractmethod
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        pass

    @abstractmethod
    def requires_incompatibility_check(self) -> bool:
        pass

    def _collect_clustering_statistics(self, clients: List[Client], *groups: Tuple[str, Any]):
        self._statistics.update({"classname": self.__class__.__name__})
        if "clients" not in self._statistics:
            self._statistics.update({"clients": {c.client_id: c.num_ex_per_class() for c in clients}})
        self._statistics.update(dict(groups))

    def requires_clients_evaluation(self) -> bool:
        return False

    def uses_custom_metric(self) -> bool:
        return False

    def _verify_constraints(self, to_empty: ClientCluster, clusters: List[ClientCluster]):
        # verify that is possible to redistribute without violating the constraint
        num_clients = np.sum([c.num_clients() for c in clusters]) + to_empty.num_clients()
        assert math.ceil(num_clients / len(clusters)) <= self._max_clients, "Conflicting constraints"

    def _redistribute_clients(self, to_empty: ClientCluster, clusters: List[ClientCluster]):
        clusters_dim_sorted = list(filter(lambda c: c.num_clients() < self._max_clients, clusters))
        clusters_dim_sorted.sort(key=lambda c: c.num_examples())
        for cluster in it.cycle(clusters_dim_sorted):
            if cluster.num_clients() < self._max_clients:
                client, client_confidence = to_empty.pop()
                cluster.add_client(client, client_confidence)
            if to_empty.num_clients() == 0:
                break

    def _check_redistribution(self, to_empty: ClientCluster, clusters: List[ClientCluster]):
        if to_empty.num_clients() > 0:
            if to_empty.num_examples() < self._min_examples:
                self._verify_constraints(to_empty, clusters)
                self._redistribute_clients(to_empty, clusters)
            else:
                clusters.append(to_empty)

    # return the number of superclient that can be build taking into account the constraints
    def _K(self, num_clients, dataset_dim) -> int:
        examples_per_client = dataset_dim // num_clients
        client_per_superclient = math.ceil(self._min_examples / examples_per_client)
        assert client_per_superclient <= self._max_clients, "Constraint infeasible: max_clients_superclient"
        num_superclients: int = num_clients // client_per_superclient
        return num_superclients

    def _save_tsne(self, clients, representers, superclients):
        if self._save_visualization and representers is not None:
            if  isinstance(clients[0].dataloader.dataset, CifarLocalDataset) and all((len(np.unique(c.dataloader.dataset.labels))==1) for c in clients) or \
                    isinstance(clients[0].dataloader.dataset, ShakespeareLocalDataset):
                representers = np.array(representers)
                clients_superclients = np.zeros(len(clients))
                for s in superclients:
                    for c in s.clients:
                        clients_superclients[c.client_id] = s.client_id
                if representers.shape[1] > 50:
                    reducer = PCA(n_components=50, svd_solver='full')
                    representers = reducer.fit_transform(representers)
                X = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(representers)
                n_subplots = 3 if isinstance(clients[0].dataloader.dataset, CifarLocalDataset) and clients[0].dataloader.dataset.num_classes == 100 else 2
                (fig, subplots) = plt.subplots(n_subplots, figsize=(15, 15))
                ax = subplots[0]
                if isinstance(clients[0].dataloader.dataset, CifarLocalDataset):
                    clients_class = np.array([c.dataloader.dataset.labels[0] for c in clients])
                    ax.set_title("representers vs single class seen")
                    colors = cm.rainbow(np.linspace(0, 1, clients[0].dataloader.dataset.num_classes))
                    for class_id, color in zip(range(clients[0].dataloader.dataset.num_classes), colors):
                        ids_of_class = np.array([True if cc == class_id else False for cc in clients_class])
                        ax.scatter(X[ids_of_class, 0], X[ids_of_class, 1], color=color)
                        ax.xaxis.set_major_formatter(NullFormatter())
                        ax.yaxis.set_major_formatter(NullFormatter())
                    ax.axis("tight")
                elif isinstance(clients[0].dataloader.dataset, ShakespeareLocalDataset):
                    clients_operas = list(json.load(open('./datasets/Shakespeare/all_data_train.json'))['hierarchies'])
                    assert len(clients_operas) == len(representers), f"Visualization available only for the full Shakespeare dataset (K={len(clients_operas)})."
                    unique_operas = list(set(clients_operas))
                    ax.set_title("representers vs opera they belong to")
                    colors = cm.rainbow(np.linspace(0, 1, len(unique_operas)))
                    for opera, color in zip(unique_operas, colors):
                        ids_of_opera = np.array([True if o == opera else False for o in clients_operas])
                        ax.scatter(X[ids_of_opera, 0], X[ids_of_opera, 1], color=color)
                        ax.xaxis.set_major_formatter(NullFormatter())
                        ax.yaxis.set_major_formatter(NullFormatter())
                    ax.axis("tight")
                ax = subplots[1]
                ax.set_title("representers vs superclient assigned to")
                colors = cm.rainbow(np.linspace(0, 1, len(superclients)))
                for superclient_id, color in zip(range(len(superclients)), colors):
                    ids_of_superclient = np.array([True if sc == superclient_id else False for sc in clients_superclients])
                    ax.scatter(X[ids_of_superclient, 0], X[ids_of_superclient, 1], color=color)
                    ax.xaxis.set_major_formatter(NullFormatter())
                    ax.yaxis.set_major_formatter(NullFormatter())
                ax.axis("tight")
                if isinstance(clients[0].dataloader.dataset, CifarLocalDataset) and clients[0].dataloader.dataset.num_classes == 100:
                    ax = subplots[2]
                    ax.set_title("representers vs superclass they belong to")
                    coarse_labels = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
                    colors = cm.rainbow(np.linspace(0, 1, 20))
                    for coarse_label, color in zip(range(20), colors):
                        ids_of_superclass = np.array([True if coarse_labels[cc] == coarse_label else False for cc in clients_class])
                        ax.scatter(X[ids_of_superclass, 0], X[ids_of_superclass, 1], color=color)
                        ax.xaxis.set_major_formatter(NullFormatter())
                        ax.yaxis.set_major_formatter(NullFormatter())
                    ax.axis("tight")
                plt.savefig(f'{self._savedir}/class_vs_superclient_viz.png')
                plt.clf()

    def _save_time_statistics(self, one_time_clustering, time, sp):
        if self._writer is not None and self._collect_time_statistics:
            if not one_time_clustering:
                self._writer.add_local_var('cluster_time', time)
                self._writer.add_local_var('n_clustering', 1)
                self._writer.add_local_var('largest_sc_#_examples', max([len(s) for s in sp]))
                self._writer.add_local_var('largest_sc_#_clients', max([len(s.clients) for s in sp]))
            else:
                self._writer.add_summary_value('largest_sc_#_examples', max([len(s) for s in sp]))
                self._writer.add_summary_value('largest_sc_#_clients', max([len(s.clients) for s in sp]))
                self._writer.add_summary_value('cluster_time', time)
                self._writer.add_summary_value('n_superclients', len(sp))

class InformedClusterMaker(ClusterMaker):
    def __init__(self, measure, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._measure = measure

    def requires_clients_evaluation(self) -> bool:
        return True

    @abstractmethod
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        pass

    @staticmethod
    def gini_diff(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        mean_vector = (cluster_vec + client_vec) / 2
        return 1 - np.sum(np.power(mean_vector, 2))

    @staticmethod
    def cosine_diff(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        v1 = cluster_vec
        v2 = client_vec
        prod = np.dot(v1, v2)
        norms_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 1 - prod / norms_prod

    @staticmethod
    def kullback_div(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        mean_vector = (cluster_vec + client_vec) / 2
        uniform = np.ones(mean_vector.size) / mean_vector.size
        klvec = [mean_vector[i] * np.log(mean_vector[i] / uniform[i]) for i in range(mean_vector.size)]
        return 1 - (np.sum(klvec))
    
    @staticmethod
    def norm_cosine_diff(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        v1 = cluster_vec / (cluster_vec + client_vec + 1e-8)
        v2 = client_vec / (cluster_vec + client_vec + 1e-8)
        return InformedClusterMaker.cosine_diff(v1, v2)

    @staticmethod
    def scipy_kullback(cluster_vec: np.ndarray, client_vec: np.ndarray) -> float:
        mean_vector = (cluster_vec + client_vec) / 2
        uniform = np.ones(mean_vector.size) / mean_vector.size
        return 1 - np.sum(rel_entr(mean_vector,uniform))

    def diff_measure(self) -> Callable[[np.ndarray, np.ndarray], float]:
        measures_methods = {"gini": InformedClusterMaker.gini_diff, "cosine": InformedClusterMaker.cosine_diff,
                            "kullback": InformedClusterMaker.kullback_div, "wasserstein": wasserstein_distance,
                            "normalized_cosine": InformedClusterMaker.norm_cosine_diff,
                            "euclidean":euclidean, "scipy_kullback": InformedClusterMaker.scipy_kullback}
        if self.measure not in measures_methods:
            raise NotImplementedError
        return measures_methods[self.measure]

    


