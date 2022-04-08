import os
import signal
from abc import ABC, abstractmethod
import time
from typing import Tuple

from src.models import Model
from src.optim import *
from torch.utils.data import DataLoader

from src.losses import *
from src.utils import exit_on_signal, MeasureMeter, savepickle
from tqdm import tqdm

class Algo(ABC):

    def __init__(self, model_info, params, device: str, dataset,
                 output_suffix: str, savedir: str, writer=None):
        Model.verify_info(model_info)
        loss_info, optim_info = params.loss, params.optim
        self.loss_fn = eval(loss_info.classname)(**loss_info.params)
        self.optimizer = eval(optim_info.classname)
        self.optimizer_args = optim_info.args

        self.device = device
        self.writer = writer
        self.dataset = dataset
        self.output_suffix = output_suffix
        self.savedir = savedir
        self._round: int = 0
        self._result = {'loss': [], 'accuracy': [], 'time_elapsed': [], "accuracy_class": []}
        self.start_time = time.time()
        self.checkpoint_path = os.path.join(self.savedir, f"checkpoint{self.output_suffix}.pkl")
        self.completed = False

        exit_on_signal(signal.SIGTERM)

    @property
    def result(self):
        return self._result

    def reset_result(self):
        self.result.update({'loss': [], 'accuracy': [], 'time_elapsed': [], "accuracy_class": []})
        self._round = 0

    @abstractmethod
    def train_step(self) -> None:
        pass

    @abstractmethod
    def validation_step(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def fit(self, num_round: int) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_from_checkpoint(self):
        pass

    @staticmethod
    def train(model: nn.Module, device: str, optimizer: nn.Module, loss_fn: nn.Module, data: DataLoader) -> None:
        model.train()
        #data.dataset.generate_mapping(4,0,1)
        for img, target in tqdm(data, desc='Training'):
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, target)

            loss.backward()
            optimizer.step()

    @staticmethod
    def test(model: nn.Module, meter: MeasureMeter, device: str, loss_fn, data: DataLoader) -> float:
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for img, target in tqdm(data, desc='Testing'):
                img = img.to(device)
                target = target.to(device)
                logits = model(img)
                test_loss += loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                meter.update(pred, target)
        test_loss = test_loss / len(data)
        return test_loss

    def save_result(self):
        savepickle(self.result, os.path.join(self.savedir, f"result{self.output_suffix}.pkl"))
