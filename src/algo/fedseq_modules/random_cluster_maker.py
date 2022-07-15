import math
from src.algo.fed_clients.base_client import Client

from src.algo.fedseq_modules.cluster_maker import ClusterMaker, ClientCluster
import copy
from typing import List
import numpy as np
import logging

log = logging.getLogger(__name__)


class RandomClusterMaker(ClusterMaker):
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        clusters: List[ClientCluster] = [] 
        s_clients = copy.copy(clients)
        np.random.shuffle(s_clients)
        if self._n_clusters == None:
            n_clusters = 0
            cluster = ClientCluster(n_clusters, logger=log)
            for c in s_clients:
                if cluster.num_examples() < self._min_examples and \
                        cluster.num_clients() < self._max_clients:
                    cluster.add_client(c)
                else:
                    clusters.append(cluster)
                    n_clusters += 1
                    cluster = ClientCluster(n_clusters, logger=log)
                    cluster.add_client(c)
            self._check_redistribution(cluster, clusters)
            self._collect_clustering_statistics(clients, ("clusters", [c.clients_id() for c in clusters]))
        else:
            n_clusters = math.floor(len(s_clients)/self._n_clusters)
            curr_id = 0
            for cluster_id in range(n_clusters):
                cluster = ClientCluster(cluster_id, logger=log)
                for c in s_clients[curr_id: curr_id + self._n_clusters]:
                    cluster.add_client(c)
                clusters.append(cluster)
                curr_id += self._n_clusters
            for i, c in enumerate(s_clients[curr_id:]):
                clusters[i].add_client(c)
        return clusters
    
    def requires_incompatibility_check(self) -> bool:
        return False