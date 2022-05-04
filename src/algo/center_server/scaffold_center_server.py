import copy
from typing import List

import torch

from src.algo.center_server.center_server import CenterServer
from src.algo.center_server.fedavg_center_server import FedAvgCenterServer
from src.algo.fed_clients import SCAFFOLDClient


class SCAFFOLDCenterServer(FedAvgCenterServer):

    def __init__(self, model, dataloader, device, num_clients: int, save_memory: bool = True):
        super().__init__(model, dataloader, device)
        # controls always in CPU
        self.server_controls = [torch.zeros_like(p.data, device="cpu") for p in self.model.parameters()
                                if p.requires_grad]
        self.num_clients = num_clients
        self.save_memory = save_memory  # if true assume clients do not modify server controls

    @staticmethod
    def from_center_server(server: CenterServer, num_clients: int):
        return SCAFFOLDCenterServer(server.model, server.dataloader, server.device, num_clients)

    def aggregation(self, clients: List[SCAFFOLDClient], aggregation_weights: List[float], round:int):
        super().aggregation(clients, aggregation_weights)
        for c in clients:
            delta_c = c.delta_controls()
            for sc, d in zip(self.server_controls, delta_c):
                sc.add_(d, alpha=1. / self.num_clients)

    def send_controls(self):
        if not self.save_memory:
            return copy.deepcopy(self.server_controls)
        return self.server_controls
