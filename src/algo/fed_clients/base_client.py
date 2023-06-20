from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
from torch.utils.data import DataLoader

from src.algo import Algo
from src.utils import MeasureMeter, generate_sigma_noise


class Client(ABC):
    def __init__(self, client_id: int, dataloader: Optional[DataLoader], savedir: str, num_classes=None, device="cpu", dp=None):
        self.client_id = client_id
        self.__dataloader = dataloader
        self.savedir =savedir
        self.device = device
        self.__model = None
        self.__num_classes = num_classes
        self.measure_meter = MeasureMeter(num_classes)
        self.dp = dp
        self.cluster_id = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        # assert isinstance(model, torch.nn.Module), "Client's model in not an instance of torch.nn.Module"
        del self.__model
        self.__model = model

    @property
    def dataloader(self) -> DataLoader:
        return self.__dataloader

    @dataloader.setter
    def dataloader(self, dataloader: DataLoader):
        assert isinstance(dataloader, DataLoader), "Client's dataloader is not an instance of torch DataLoader"
        self.__dataloader = dataloader

    @property
    def num_classes(self):
        return self.__num_classes

    @abstractmethod
    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        pass

    def client_evaluate(self, loss_fn, test_data: DataLoader) -> Tuple[float, MeasureMeter]:
        self.model.to(self.device)
        loss_fn.to(self.device)
        self.measure_meter.reset()
        loss = Algo.test(self.model, self.measure_meter, self.device, loss_fn, test_data)
        return loss, self.measure_meter

    def __len__(self):
        return len(self.dataloader.dataset)

    def num_ex_per_class(self) -> np.array:
        num_ex_per_class = np.zeros(self.num_classes)
        for _, batch in self.dataloader:
            for target in batch.numpy():
                num_ex_per_class[target] += 1
        return num_ex_per_class

    def send_model(self):
        if self.dp:
            dataset_size = len(self)
            sigma2 = generate_sigma_noise(dataset_size, self.dp.epsilon, self.dp.delta, self.dp.C)
            self.model.add_gaussian_noise(sigma2, self.dp.C)
        return self.model
