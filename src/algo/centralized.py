import logging
import pickle
import time

import torch.optim.lr_scheduler

from src.models import Model
from src.datasets import *
from torch.utils.data import DataLoader
from src.utils import MeasureMeter, savepickle
from src.utils import get_dataset
from src.algo import Algo

log = logging.getLogger(__name__)


class Centralized(Algo):
    def __init__(self, model_info, params, device: str, dataset,
                 output_suffix: str, savedir: str, writer=None, wandbConf=None):
        assert params.loss.type == "crossentropy", "Loss function for centralized algorithm must be crossentropy"
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self.batch_size = params.batch_size
        self.wandbConf = wandbConf

        dataset_getter, transform = get_dataset(dataset.name)
        dataset_class= eval(dataset.dataset_class)
        dataset_num_classes= dataset.dataset_num_class
        train_img, train_label, test_img, test_label = dataset_getter(dataset_name=dataset.name)
        train_img, train_label, test_img, test_label = transform(True, train_img, train_label, test_img, test_label)
        training_set = dataset_class(train_img, train_label, dataset_num_classes)
        test_set = dataset_class(test_img, test_label, dataset_num_classes, train=False)

        self.model = Model(model_info, dataset_num_classes)

        self.train_loader = DataLoader(training_set,
                                       num_workers=6,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        self.test_loader = DataLoader(test_set,
                                      num_workers=6,
                                      batch_size=self.batch_size,
                                      shuffle=False)
        self.measure_meter = MeasureMeter(self.model.num_classes)
        self.scheduler = None
        self.optim = None

    def fit(self, epochs):
        self.epochs = epochs
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.model.train()
        self.optim = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, epochs)
        self._round = 0
        self.validation_step()

        try:
            for i in range(epochs):
                self._round = i + 1
                self.train_step()
                self.validation_step()
            self.completed = True
        except SystemExit:
            log.warning(f"Training stopped at round {self._round}")
        finally:
            self.save_checkpoint()
            if self.completed:
                self.save_result()
                self.writer.add_summary_value(f'Average_accuracy_{self.dataset.average_accuracy_rounds}_rounds',\
                    self.writer.local_store['Avg_acc']/self.dataset.average_accuracy_rounds)


    def train_step(self):
        Algo.train(self.model, self.device, self.optim, self.loss_fn, self.train_loader)
        self.scheduler.step()

    def validation_step(self):
        self.measure_meter.reset()
        now = time.time()
        test_loss = Algo.test(self.model, self.measure_meter, self.device, self.loss_fn, self.test_loader)
        accuracy = self.measure_meter.accuracy_overall
        log.info(
            f"[Epochs: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            if self._round%self.wandbConf.centralized.tot_round==0:
                name = "model" if self.wandbConf.centralized.policy=='last' else f'model_r{self._round}'
                self.writer.save_model(self.model, name=name)
            if self.epochs-self._round<self.dataset.average_accuracy_rounds:
                self.writer.add_local_var('Avg_acc',accuracy)
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)
            self.writer.add_scalar("val/time_elapsed", now, self._round)

        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(accuracy)
        self.result['time_elapsed'].append(now)
        self.result['accuracy_class'].append(self.measure_meter.accuracy_per_class)

    def save_checkpoint(self):
        savepickle({**self.result, "round": self._round, "model": self.model},
                   self.checkpoint_path)

    def load_from_checkpoint(self):
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                assert all(key in checkpoint_data for key in self.result.keys()), "Missing data in checkpoint"
                assert "model" in checkpoint_data and "round" in checkpoint_data \
                       and isinstance(checkpoint_data["model"], Model), "Missing model"
                assert checkpoint_data["model"].same_setup(self.model)
                log.info(f'Reloading checkpoint from round {checkpoint_data["round"]}')
                for k in self.result.keys():
                    self.result[k] = checkpoint_data[k]
                self.model = checkpoint_data["model"]
                self._round = checkpoint_data["round"]
        except BaseException as err:
            log.warning(f"Unable to load from checkpoint, starting from scratch: {err}")


