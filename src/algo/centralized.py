import logging
import pickle
import time
import numpy as np
import torch.optim.lr_scheduler
from torch.nn.modules.loss import CrossEntropyLoss
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
        train_img, train_label, test_img, test_label = dataset_getter(dataset_name=dataset.name, device=self.device)
        train_img, train_label, test_img, test_label = transform(True, train_img, train_label, test_img, test_label)
        training_set = dataset_class(train_img, train_label, dataset_num_classes, device)
        test_set = dataset_class(test_img, test_label, dataset_num_classes, device, train=False)

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

    def fit(self, epochs: int):
        self.epochs = epochs
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.model.train()
        self.optim = self.optimizer(self.model.parameters(), **self.optimizer_args)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, epochs)
        if self._round==0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, epochs)
            self.validation_step()
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, epochs, last_epoch=self._round)

        try:
            for i in range(self._round, epochs):
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
                if isinstance(self.test_loader.dataset, StackoverflowLocalDataset):
                    loss = Algo.test(self.model, self.measure_meter, self.device, CrossEntropyLoss(), self.test_loader)
                    accuracy = self.measure_meter.accuracy_overall
                    self.writer.add_summary_value('Final_loss_whole_dataset', loss)
                    self.writer.add_summary_value('Final_accuracy_whole_dataset', accuracy)
                self.writer.add_summary_value(f'Average_accuracy_{self.dataset.average_accuracy_rounds}_rounds',\
                    self.writer.local_store['Avg_acc']/self.dataset.average_accuracy_rounds)


    def train_step(self):
        Algo.train(self.model, self.device, self.optim, self.loss_fn, self.train_loader)
        self.scheduler.step()

    def validation_step(self):
        self.measure_meter.reset()
        now = time.time()
        if isinstance(self.test_loader.dataset, StackoverflowLocalDataset):
            random_indices = set(np.random.default_rng().choice(len(self.test_loader)-1, int(10000 / self.test_loader.batch_size), replace=False))
            test_loss = Algo.test_subsample(self.model, self.measure_meter, self.device, self.loss_fn, self.test_loader, random_indices)    
        else:
            test_loss = Algo.test(self.model, self.measure_meter, self.device, self.loss_fn, self.test_loader)
        accuracy = self.measure_meter.accuracy_overall
        log.info(
            f"[Epochs: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.save_model(self.model, 'last_model', self._round)
            if self.wandbConf.server_model.save_every_n_rounds:
                if self._round%self.wandbConf.server_model.tot_round==0:
                    self.writer.save_model(self.model, f'model_r{self._round}', self._round)
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
            log.info(f'Reloading checkpoint from round {self.writer.restore_run["resume_round"]}')
            model_weight = self.writer.restore_run["model_weight"]
            model_weight = {'.'.join(k.split('.')[1:]): v for k,v in model_weight.items()}
            self.model.model.load_state_dict(model_weight)
            self._round = self.writer.restore_run["resume_round"]
        except BaseException as err:
            log.warning(f"Unable to load from checkpoint, starting from scratch: {err}")