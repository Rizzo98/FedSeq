import torch
from typing import List
from src.algo.center_server import FedAvgCenterServer, CenterServer
from src.algo.fed_clients import Client


class FedDynCenterServer(FedAvgCenterServer):
    def __init__(self, model, dataloader, device, alpha, num_clients):
        super().__init__(model, dataloader, device)
        self.h = [torch.zeros_like(p.data, device=device) for p in self.model.parameters()]
        self.alpha = alpha
        self.num_clients = num_clients

    @staticmethod
    def from_center_server(server: CenterServer, alpha, num_clients):
        return FedDynCenterServer(server.model, server.dataloader, server.device, alpha, num_clients)

    def aggregation(self, clients: List[Client], aggregation_weights: List[float]):
        # compute the sum of all the model parameters of the clients involved in training
        sum_theta = [torch.zeros_like(p.data) for p in self.model.parameters()]
        for c in clients:
            for s, c_theta in zip(sum_theta, c.model.parameters()):
                s.add_(c_theta)
        # compute the deltas w.r.t. the old server model
        delta_theta = [torch.clone(p) for p in sum_theta]
        num_participating_clients = len(clients)
        for d, p in zip(delta_theta, self.model.parameters()):
            d.add_(p.data, alpha=-num_participating_clients)
        # update the h parameter
        for h, theta in zip(self.h, delta_theta):
            h.data.add_(theta, alpha=-(self.alpha / self.num_clients))
        # update the server model
        for model_param, h, sum_theta_p in zip(self.model.parameters(), self.h, sum_theta):
            model_param.data = 1. / len(clients) * sum_theta_p.data - 1. / self.alpha * h.data
