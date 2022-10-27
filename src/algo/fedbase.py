import datetime
import os
from torch.nn.modules.loss import CrossEntropyLoss
import time
import numpy as np
from abc import abstractmethod
from torch.utils.data import DataLoader
from src.datasets import StackoverflowLocalDataset
from src.algo.center_server import *
from src.utils import create_datasets, savepickle
from src.algo import Algo
from src.algo.fed_clients import *
from src.models import Model
import logging

log = logging.getLogger(__name__)


class FedBase(Algo):
    def __init__(self, model_info, params, device: str, dataset,
                 output_suffix: str, savedir: str, writer=None, wandbConf=None ,*args, **kwargs):
        common = params.common
        C, K, B, E, alpha = common.C, common.K, common.B, common.E, common.alpha
        assert 0 < C <= 1, f"Illegal value, expected 0 < C <= 1, given {C}"
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer)
        self.num_clients = K
        self.batch_size = B
        self.fraction = C
        self.local_epoch = E
        self.dp = common.dp
        self.alpha = alpha
        self.dataset_num_classes = dataset.dataset_num_class
        self.aggregation_policy = params.aggregation_policy
        self.save_checkpoint_period = params.save_checkpoint_period
        self.wandbConf = wandbConf
        self.accuracy_centralized = dataset.accuracy_centralized
        self.percentages_saved = dataset.percentages_saved
        self.epochs_avg_accuracy = 5
        # get the proper dataset
        self.local_datasets, self.test_dataset = create_datasets(self.dataset, self.num_clients, alpha, device=self.device)
        self.model_info = model_info
        model = Model(model_info, self.dataset_num_classes)
        model_has_batchnorm = model.has_batchnorm()
        local_dataloaders = [
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=model_has_batchnorm)
            for dataset in self.local_datasets]

        self.clients = [
            eval(params.client.classname)(k, local_dataloaders[k], self.dataset_num_classes, self.device,
                                          self.dp, **params.client.args) for k in range(self.num_clients)
        ]
        self.selected_clients = []

        # take out examplars from test_dataset, will be used in FedSeq
        original_test_set_len = len(self.test_dataset)
        self.excluded_from_test = self.test_dataset.get_subset_eq_distr(self.dataset.num_exemplar)
        log.info(f"Len of total test set = {original_test_set_len}")
        log.info(
            f"Len of reduced test set = {len(self.test_dataset)}, {100 * len(self.test_dataset) / original_test_set_len}% of total test set")
        log.info(f"Len of extracted examples from test set = {len(self.excluded_from_test)}")

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        self.center_server = eval(params.center_server.classname)(model, self.test_dataloader, self.device,
                                                                  **params.center_server.args)
        self.save_models = params.save_models

    @abstractmethod
    def train_step(self):
        pass

    def validation_step(self):
        test_loss, meter = self.center_server.validation(CrossEntropyLoss())
        accuracy = meter.accuracy_overall
        now = time.time()
        log.info(
            f"[Round: {self._round: 05}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.save_model(self.center_server.model, 'last_model', self._round)
            if self.wandbConf.server_model.save_every_n_rounds:
                if self._round%self.wandbConf.server_model.tot_round==0:
                    self.writer.save_model(self.center_server.model, f'model_r{self._round}', self._round)
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)
            self.writer.add_scalar("val/time_elapsed", now, self._round)
            if self.num_round-self._round<self.dataset.average_accuracy_rounds:
                self.writer.add_local_var('Avg_acc',accuracy)
            if self._round == 0:
                    self.writer.add_local_var('Last_rounds_accuracies', [-1]*self.epochs_avg_accuracy)
                    self.writer.add_local_var('index_passed_percentage', 0)
            if self.accuracy_centralized != None and self.writer.local_store['index_passed_percentage'] < len(self.percentages_saved):
                self.writer.local_store['Last_rounds_accuracies'][self._round % self.epochs_avg_accuracy] = accuracy.item()
                self.writer.save_object(self.writer.local_store, 'local_store')
                last_rounds_avg = np.average(list(filter(lambda x: x!=-1, self.writer.local_store['Last_rounds_accuracies'])))
                if last_rounds_avg > self.percentages_saved[self.writer.local_store['index_passed_percentage']]*self.accuracy_centralized:
                    self.writer.add_summary_value(f"round_over_{self.percentages_saved[self.writer.local_store['index_passed_percentage']]}_acc", self._round)   
                    self.writer.add_local_var('index_passed_percentage', 1)
        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(accuracy)
        self.result['time_elapsed'].append(now)
        self.result['accuracy_class'].append(meter.accuracy_per_class)

    def fit(self, num_round: int):
        self.num_round = num_round
        if self.wandbConf.restart_from_run is None:
            if self.wandbConf.client_datasets:
                bins = []
                for c in self.clients: bins.append([c.client_id]+list(c.num_ex_per_class()))
                #bins = [list(np.bincount(c.labels, minlength=self.dataset.dataset_num_class)) for c in self.local_datasets]
                self.writer.add_table(bins,['Client_id']+[f'Class {j}' for j in range(len(bins[0])-1)],'Clients distribution')
            self.validation_step()

        self.save_clients_model()
        if self.completed:
            self.reset_result()
        
        try:
            for t in range(self._round, num_round):
                self.train_step()
                self._round = t + 1
                self.aggregate(self.selected_clients)
                if self._round % self.save_checkpoint_period == 0:
                    self.save_checkpoint()
                self.validation_step()
                self.reset_clients_model()
            self.completed = True
        except SystemExit:
            log.warning(f"Training stopped at round {self._round}")
        finally:
            self.save_checkpoint()
            if self.completed:
                self.save_result()
                if isinstance(self.center_server.dataloader.dataset, StackoverflowLocalDataset):
                    loss = Algo.test(self.center_server.model, self.center_server.measure_meter, self.center_server.device, CrossEntropyLoss(), self.center_server.dataloader)
                    accuracy = self.center_server.measure_meter.accuracy_overall
                    self.writer.add_summary_value('Final_loss_whole_dataset', loss)
                    self.writer.add_summary_value('Final_accuracy_whole_dataset', accuracy)
                    log.info(
                        f"[Final round] Test set on whole Stackoverflow dataset: Average loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
                    )
                self.writer.add_summary_value(f'Average_accuracy_{self.dataset.average_accuracy_rounds}_rounds',\
                    self.writer.local_store['Avg_acc']/self.dataset.average_accuracy_rounds)
                if hasattr(self, 'avg_n_superclients'):
                    self.writer.add_summary_value('avg_n_superclients', self.avg_n_superclients)
                if hasattr(self, 'clustering_method') and self.clustering.collect_time_statistics:
                    self.writer.add_summary_value('total_cluster_time', str(datetime.timedelta(seconds=(self.clustering_method._writer.local_store['cluster_time']))))
                    self.writer.add_summary_value('avg_cluster_time', str(datetime.timedelta(seconds=(self.clustering_method._writer.local_store['cluster_time']/self.clustering_method._writer.local_store['n_clustering']))))
                    self.writer.add_summary_value('#_times_clustering', self.clustering_method._writer.local_store['n_clustering'])
                    self.writer.add_summary_value('avg_largest_sc_#_examples', self.avg_largest_sc_n_examples/self.num_round)
                    self.writer.add_summary_value('avg_largest_sc_#_clients', self.avg_largest_sc_n_clients/self.num_round)
    def save_clients_model(self):
        if self.save_models:
            for c in self.clients:
                savepickle(c.model, os.path.join(self.savedir, f"models{self.output_suffix}", f"{c.client_id}.pkl"))

    def load_from_checkpoint(self):
        try:
            log.info(f'Reloading checkpoint from round {self.writer.restore_run["resume_round"]}')
            model_weight = self.writer.restore_run["model_weight"]
            model_weight = {'.'.join(k.split('.')[1:]): v for k,v in model_weight.items()}
            self.center_server.model.model.load_state_dict(model_weight)
            self._round = self.writer.restore_run["resume_round"]
        except BaseException as err:
            log.warning(f"Unable to load from checkpoint, starting from scratch: {err}")

    def send_data(self, clients):
        for client in clients:
            client.model = self.center_server.send_model()

    def reset_clients_model(self):
        for c in self.selected_clients:
            c.model = None

    def aggregate(self, clients):
        if self.aggregation_policy == "weighted":
            total_weight = sum([len(c) for c in clients])
            weights = [len(c) / total_weight for c in clients]
        else:  # uniform
            total_weight = len(clients)
            weights = [1. / total_weight for _ in range(len(clients))]
        self.center_server.aggregation(clients, weights, self._round)

    def save_checkpoint(self):
        savepickle({**self.result, "round": self._round, "model": self.center_server.model},
                   self.checkpoint_path)
