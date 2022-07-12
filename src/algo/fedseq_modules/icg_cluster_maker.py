import math
import random
from src.algo.fed_clients.base_client import Client

from src.algo.fedseq_modules.cluster_maker import InformedClusterMaker, ClientCluster
from typing import List
import numpy as np
import math
from scipy.spatial.distance import cdist
from ortools.graph import pywrapgraph
import logging

log = logging.getLogger(__name__)


class ICGClusterMaker(InformedClusterMaker):
    def __init__(self, n_max_iterations: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_max_iterations = n_max_iterations

    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray]) -> List[ClientCluster]:
        self.n_sampled_clients = self._n_clusters * math.floor(len(clients) / self._n_clusters)
        self.n_clients_x_cluster = int(self.n_sampled_clients / self._n_clusters)
        client_representers = list(zip(clients, representers))
        random.shuffle(client_representers)
        client_representers = list(zip(*client_representers))
        self.sampled_clients = list(client_representers[0])[:self.n_sampled_clients]
        self.sampled_representers = np.array(list(client_representers[1])[:self.n_sampled_clients])
        self._populate_assignment_matrix()
        iteration = 0
        while iteration == 0 or (iteration < self.n_max_iterations and self._obtainedDifferentAssignment()):
            centroids = self._fix_centroids()
            costs = self._calculate_costs(centroids)
            self._solve_assignment_problem(costs)
            iteration += 1
        clusters = self._obtain_clusters()
        groups = self._sample_from_clusters(clusters)
        return groups

    def _fix_centroids(self):
        centroids : List[np.ndarray] = []
        for cluster in range(self._n_clusters):
            centroids.append((np.sum(self.sampled_representers[(self.assigned_clients[cluster,:] == 1), :], axis=0) / self.n_clients_x_cluster)[np.newaxis, :])
        return centroids

    def _calculate_costs(self, centroids):
        costs = np.zeros((self._n_clusters, self.n_sampled_clients))
        for i, centroid in enumerate(centroids):
            costs[i] = cdist(centroid, self.sampled_representers, metric='sqeuclidean')
        exponent = math.floor(math.log10(np.min(costs)))
        costs = 10**(-exponent)*costs
        return costs

    def _solve_assignment_problem(self, costs):
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        clients = [i for i in range(self.n_sampled_clients)]
        clusters = [i for i in range(self.n_sampled_clients, self.n_sampled_clients + self._n_clusters)]
        for i, client in enumerate(clients):
            for j, cluster in enumerate(clusters):
                min_cost_flow.AddArcWithCapacityAndUnitCost(client, cluster, 1, int(costs[j, i]))
            min_cost_flow.SetNodeSupply(client, 1)
        for cluster in clusters:
            min_cost_flow.SetNodeSupply(cluster, -self.n_clients_x_cluster)
        status = min_cost_flow.Solve()
        if status != min_cost_flow.OPTIMAL:
            print('There was an issue with the min cost flow input.')
            print(f'Status: {status}')
            exit(1)
        self._populate_assignment_matrix(solver_results=min_cost_flow)
        
    def _populate_assignment_matrix(self, solver_results=None):
        self.previous_assigned_clients = np.zeros((self._n_clusters, self.n_sampled_clients))
        if solver_results is None:
            self.assigned_clients = np.zeros((self._n_clusters, self.n_sampled_clients))
            available_clients = set(range(self.n_sampled_clients))        
            for cluster in range(self._n_clusters):
                selected_clients = np.random.choice(list(available_clients), self.n_clients_x_cluster, replace=False)
                for selected_client in selected_clients:
                    available_clients.remove(selected_client)
                    self.assigned_clients[cluster, selected_client] = 1
        else:
            self.previous_assigned_clients = self.assigned_clients
            self.assigned_clients = np.zeros((self._n_clusters, self.n_sampled_clients))
            for i in range(solver_results.NumArcs()):
                if solver_results.Flow(i) > 0:
                    self.assigned_clients[solver_results.Head(i) - self.n_sampled_clients, solver_results.Tail(i)] = 1
     
    def _obtain_clusters(self):
        clusters = {i: [] for i in range(self._n_clusters)}
        for i in range(self._n_clusters):
            clusters[i] = list(np.where(self.assigned_clients[i,:] == 1)[0])
        return clusters
    
    def _sample_from_clusters(self, clusters):
        groups : List[ClientCluster] = []
        for g in range(self.n_clients_x_cluster):
            group = ClientCluster(g, logger=log)
            for cluster in range(self._n_clusters):
                idx = int(np.random.choice(range(len(clusters[cluster])), 1))
                client = clusters[cluster].pop(idx)
                group.add_client(self.sampled_clients[client])
            groups.append(group)
        return groups

    def _obtainedDifferentAssignment(self):
        return not np.array_equal(self.assigned_clients, self.previous_assigned_clients)

    
    def requires_incompatibility_check(self) -> bool:
        return False