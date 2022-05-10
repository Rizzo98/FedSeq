import copy
from src.algo.fedseq_modules import FedSeqSuperClient
from src.algo.fed_clients import Client
from typing import List
from src.models import Model
from src.algo.center_server import FedAvgCenterServer
import networkx as nx

class GraphSuperclient(FedSeqSuperClient):
    def __init__(self, superclient_id, clients: List[Client], clients_local_epoch: int, num_classes: int, average_model: bool, check_forgetting: bool, shuffle_sp_clients: bool, clients_dropout: float, *args, **kwargs):
        super().__init__(superclient_id, clients, clients_local_epoch, num_classes, average_model, check_forgetting, shuffle_sp_clients, clients_dropout, *args, **kwargs)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.clients)
        for n1 in self.graph.nodes:
            for n2 in self.graph.nodes:
                if n1!=n2:
                    self.graph.add_edge(n1,n2, weight=1/(self.graph.number_of_nodes()-1))

    def client_update(self, optimizer, optimizer_args, sequential_rounds, loss_fn):
        model_sending = self._select_sending_strategy()        
        for c in self.clients:
            model_sending(c, self.model)

        step=10
        trained_models = [copy.deepcopy(self.model) for _ in range(len(self.clients))]
        for _ in range(step):
            for c in self.clients:
                self._train_single_client(c, optimizer, optimizer_args, self.clients_local_epoch, loss_fn)
            for i,c in enumerate(self.clients):
                neigh_models = [neigh.model for neigh in list(self.graph.neighbors(c))]
                weights = [self.graph.edges[(c,neigh)]['weight'] for neigh in list(self.graph.neighbors(c))]
                FedAvgCenterServer.weighted_aggregation(neigh_models,weights,trained_models[i])
            for i,c in enumerate(self.clients):
                c.model = trained_models[i]

        FedAvgCenterServer.weighted_aggregation([c.model for c in self.clients], [1/len(self.clients) for _ in self.clients], self.model)