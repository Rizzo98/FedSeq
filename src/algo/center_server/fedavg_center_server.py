from collections import OrderedDict
from typing import Iterable, List, Tuple
import numpy as np
from src.algo import Algo
from src.algo.center_server.center_server import CenterServer
from src.algo.fed_clients import Client
from src.models import Model
from src.utils import MeasureMeter
from src.datasets import StackoverflowLocalDataset


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device):
        super().__init__(model, dataloader, device)

    @staticmethod
    def weighted_aggregation(models: Iterable[Model], aggregation_weights: List[float], dest: Model):
        update_state = OrderedDict()

        for k, model in enumerate(models):
            local_state = model.state_dict()
            for key in model.state_dict().keys():
                if k == 0:
                    update_state[
                        key] = local_state[key] * aggregation_weights[k]
                else:
                    update_state[
                        key] += local_state[key] * aggregation_weights[k]
        dest.load_state_dict(update_state)

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], round:int):
        FedAvgCenterServer.weighted_aggregation([c.send_model() for c in clients], aggregation_weights, self.model)

    def validation(self, loss_fn) -> Tuple[float, MeasureMeter]:
        self.model.to(self.device)
        loss_fn.to(self.device)
        self.measure_meter.reset()
        if isinstance(self.dataloader.dataset, StackoverflowLocalDataset):
            random_indices = set(np.random.default_rng().choice(len(self.dataloader)-1, int(10000 / self.dataloader.batch_size), replace=False))
            loss = Algo.test_subsample(self.model, self.measure_meter, self.device, loss_fn, self.dataloader, random_indices)
        else:
            loss = Algo.test(self.model, self.measure_meter, self.device, loss_fn, self.dataloader)
        return loss, self.measure_meter
