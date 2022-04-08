from typing import List
import numpy as np
from src.algo.fed_clients.base_client import Client
import pulp as pl
from src.algo.fedseq_modules.cluster_maker import ClientCluster, InformedClusterMaker
from src.algo.fedseq_modules import FedSeqSuperClient
import logging

log = logging.getLogger(__name__)


class ConfidenceILP(InformedClusterMaker):
    def _make_clusters(self, clients: List[Client], representers: List[np.ndarray], average_model: bool, clients_local_epoch: int) -> \
            List[ClientCluster]:
        clients_size, num_clients, num_superclients, homogeneity_matrix = self.make_data(clients, representers)
        while True:
            log.info(f"Attempting ILP with num_superclients={num_superclients}")
            ilp = self.make_ilp(clients_size, num_clients, num_superclients, homogeneity_matrix)
            ilp.solve(pl.PULP_CBC_CMD(msg=False))
            if ilp.status == pl.constants.LpStatusOptimal:
                break
            del ilp
            num_superclients -= 1
            assert num_superclients > 0, "Infeasible"
        x = [bool(v.varValue) for k, v in ilp.variablesDict().items() if k.find("superclient") != -1]
        superclient_masks = np.array_split(x, num_superclients)
        superclients = []
        clients_array = np.array(clients)
        for superclient_id, mask in enumerate(superclient_masks):
            sp = FedSeqSuperClient(superclient_id, clients_array[mask], clients_local_epoch, average_model)
            log.info(f"superclient {sp.client_id}: examples: {len(sp)}, clients: {sum(mask)}")
            superclients.append(sp)
        return superclients

    def make_data(self, clients: List[Client], predictions: List[np.ndarray]):
        num_clients = len(clients)
        clients_size = [len(client) for client in clients]
        dataset_dim = sum(clients_size)
        num_superclients: int = self._K(num_clients, dataset_dim)

        # homogeneity matrix: h[i][j] := homogeneity of client i and client j
        homogeneity_matrix = np.empty((num_clients, num_clients))
        np.fill_diagonal(homogeneity_matrix, 0.000000001)
        measure = self.diff_measure()
        for i in range(num_clients):
            for j in range(i):
                homogeneity_matrix[i][j] = homogeneity_matrix[j][i] = measure(predictions[i], predictions[j])

        return clients_size, num_clients, num_superclients, homogeneity_matrix

    def make_ilp(self, clients_size, num_clients, num_superclients, homogeneity_matrix):
        model = pl.LpProblem("Superclient_partition", pl.LpMaximize)
        x = pl.LpVariable.matrix("client in superclient", (range(num_clients), range(num_superclients)),
                                 lowBound=0,
                                 upBound=1,
                                 cat='Integer')
        # indicator variable p := p_ijk = 1 => client i and j are in the same cluster k
        p = pl.LpVariable.matrix("clients in same cluster",
                                 (range(num_clients), range(num_clients), range(num_superclients)),
                                 lowBound=0,
                                 upBound=1,
                                 cat='Integer')
        # indicator variable v := v_k = 1 => cluster k is void
        v = pl.LpVariable.dicts("cluster is void", range(num_superclients), 0, 1, 'Integer')

        # bounding p to x: if x_ik==0 or x_jk==0 => p_ijk = 0
        # the formulation of the problem (max) will try to put p_ijk = 1, so the constraint
        # should only ensure it is not when x_ik==0 or x_jk==0
        # constraints = [p[i][j][k]-0.5*(x[i][k]+x[j][k])<=0 for i in range(num_clients) for j in range(i) for k in range(num_superclients)]
        for k in range(num_superclients):
            for i in range(num_clients):
                for j in range(i):
                    model += p[i][j][k] <= 0.5 * x[i][k] + 0.5 * x[j][k]

        # bounding v to p
        for k in range(num_superclients):
            model += v[k] >= 1 - pl.lpSum([p[i][j][k] for i in range(num_clients) for j in range(i)])
            model += self._max_clients * (1 - v[k]) >= pl.lpSum(
                [p[i][j][k] for i in range(num_clients) for j in range(i)])
            model += pl.lpSum([(clients_size[i] + clients_size[j]) * p[i][j][k]] for i in range(num_clients) for j in
                              range(i)) >= 2 * self._min_examples * (1 - v[k])

        intra_cluster_homogeneity = [homogeneity_matrix[i][j] * p[i][j][k] for i in range(num_clients) for j in range(i)
                                     for k in range(num_superclients)]
        model += pl.lpSum(intra_cluster_homogeneity), "Obj: maximize homogeneity of cluster"

        # Constraint 1 - each client is not in more than one superclient
        for i in range(num_clients):
            model += pl.lpSum([x[i][j] for j in range(num_superclients)]) == 1

        # #Constraint 2 - each superclient has at least a minimum number of examples
        # for j in range(num_superclients):
        #     model += pl.lpSum([clients_size[i]*x[i][j] for i in range(num_clients)])>=self._min_num_examples

        # Constraint 3 - each superclient is composed by no more than a max number of clients
        # for j in range(num_superclients):
        #     model += pl.lpSum([x[i][j] for i in range(num_clients)])<=self._max_clients_superclient
        # for k in range(num_superclients):
        #     model += pl.lpSum([(clients_size[i]+clients_size[j])*p[i][j][k] for i in range(num_clients) for j in range(i)]) <= self._max_clients_superclient/2*(1-v[k])
        return model
