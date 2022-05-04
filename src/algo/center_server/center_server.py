import copy
from typing import List
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from src.algo.fed_clients.base_client import Client
from src.models import Model
from src.utils import MeasureMeter


class CenterServer(ABC):
    def __init__(self, model: Model, dataloader: DataLoader, device: str):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.measure_meter = MeasureMeter(model.num_classes)

    @abstractmethod
    def aggregation(self, clients: List[Client], aggregation_weights: List[float], round: int):
        pass

    def send_model(self):
        return copy.deepcopy(self.model)

    @abstractmethod
    def validation(self, loss_fn):
        pass
