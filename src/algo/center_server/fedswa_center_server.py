from typing import List
from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import Client
from typing import Tuple
import numpy as np
from src.algo import Algo
from src.utils import MeasureMeter
from src.datasets import StackoverflowLocalDataset

class FedSWACenterServer(FedAvgCenterServer):
    def __init__(self, model, dataloader, device, num_clients, c, tot_epochs):
        super().__init__(model, dataloader, device)
        self.num_clients = num_clients
        self.c = c
        self.swa_start = 0.75 * tot_epochs
        self.swa_n = 0 
        self.swa_model = None

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], round:int):
        super().aggregation(clients, aggregation_weights, round)
        if round > self.swa_start and (round - self.swa_start) % self.c == 0:
            alpha = (1.0 / (self.swa_n + 1))
            for param1, param2 in zip(self.swa_model.parameters(), self.model.parameters()):
                param1.data *= (1.0 - alpha)
                param1.data += param2.data * alpha
            self.swa_n += 1

    def validation(self, loss_fn) -> Tuple[float, MeasureMeter]:
        model = self.model if self.swa_model is None else self.swa_model
        model.to(self.device)
        loss_fn.to(self.device)
        self.measure_meter.reset()
        if isinstance(self.dataloader.dataset, StackoverflowLocalDataset):
            random_indices = set(np.random.default_rng().choice(len(self.dataloader)-1, int(10000 / self.dataloader.batch_size), replace=False))
            loss = Algo.test_subsample(model, self.measure_meter, self.device, loss_fn, self.dataloader, random_indices)
        else:
            loss = Algo.test(model, self.measure_meter, self.device, loss_fn, self.dataloader)
        return loss, self.measure_meter

