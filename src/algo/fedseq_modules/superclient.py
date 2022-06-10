from typing import List, Callable
import numpy as np
from src.utils import shuffled_copy, select_random_subset
from torch.utils.data import DataLoader
from src.algo.fed_clients import Client
import copy
import logging
from src.algo.center_server.fedavg_center_server import FedAvgCenterServer
from src.models import Model
from torch.nn.modules.loss import CrossEntropyLoss

log = logging.getLogger(__name__)


class FedSeqSuperClient(Client):
    def __init__(self, superclient_id, clients: List[Client], clients_local_epoch: int, num_classes: int,
                 average_model: bool, check_forgetting: bool, shuffle_sp_clients: bool, clients_dropout: float,
                 *args, **kwargs):
        super().__init__(superclient_id, None, num_classes)
        self.__clients = clients
        self.__average_model = average_model
        self.__check_forgetting = check_forgetting
        self.clients_local_epoch = clients_local_epoch
        self.__shuffle_sp_clients = shuffle_sp_clients
        self.__clients_dropout = clients_dropout
        self.__forgetting_stats = []

    @property
    def forgetting_stats(self):
        return self.__forgetting_stats

    @property
    def clients(self):
        return self.__clients
    
    @clients.setter
    def clients(self,vals):
        assert len(vals)==2, 'Wrong number of args'
        pos = vals[0]
        c = vals[1]
        assert pos>=0 and pos<len(self.__clients),'Incorrect position'
        assert isinstance(c,Client),'Incorrect client'
        self.__clients[pos]=c

    def __len__(self):
        return sum([len(client) for client in self.__clients])

    def __get_model_from_client(self, client: Client):
        return client.send_model()
        # add delay here to simulate real life

    def _train_single_client(self, client: Client, optimizer, optimizer_args, local_epoch, loss_fn):
        client.client_update(optimizer, optimizer_args, local_epoch, loss_fn)

    def __compute_averaged(self, models: List[Model]) -> None:
        tot_len = len(self)
        weights = [len(client) / tot_len for client in self.__clients]
        FedAvgCenterServer.weighted_aggregation(models, weights, self.model)

    def __take_last_trained(self, models: List[Model]) -> None:
        self.model = models[-1]

    def client_update(self, optimizer, optimizer_args, sequential_rounds, loss_fn):
        self.__forgetting_stats = []
        dropping = self.__clients_dropping()
        clients_ordering = self.__select_clients_ordering()
        model_sending = self._select_sending_strategy()
        final_model_computing: Callable[[List[Model]], None] = self.__select_model_computing()
        for _ in range(sequential_rounds):
            models, current_model, ordered_clients = [], self.model, clients_ordering(dropping(self.__clients))
            self.__forgetting_stats.append({})
            for client in ordered_clients:
                model_sending(client, current_model)
                self._train_single_client(client, optimizer, optimizer_args, self.clients_local_epoch, loss_fn)
                self.__check_catastrophic_forgetting(client, CrossEntropyLoss(), ordered_clients)
                current_model = self.__get_model_from_client(client)
                models.append(current_model)
            final_model_computing(models)

    def __select_clients_ordering(self):
        if self.__shuffle_sp_clients:
            return shuffled_copy
        else:
            return lambda x: x

    def __check_catastrophic_forgetting(self, c: Client, loss_fn, ordered_clients: List[Client]):
        if self.__check_forgetting:
            forgetting = []
            for prev_client in ordered_clients:
                d = prev_client.dataloader
                dataset, batch_size, drop_last, num_workers = d.dataset, d.batch_size, d.drop_last, d.num_workers
                test_dataloader = DataLoader(dataset, batch_size, False, drop_last=drop_last, num_workers=num_workers)
                loss, meter = c.client_evaluate(loss_fn, test_dataloader)
                forgetting.append({"id": prev_client.client_id, "loss": loss, "accuracy": meter.accuracy_overall})
                if prev_client.client_id == c.client_id:
                    break
            self.__forgetting_stats[-1][c.client_id] = forgetting

    def num_ex_per_class(self) -> np.array:
        ex_per_class = np.zeros(self.num_classes)
        for client in self.__clients:
            ex_per_class += client.num_ex_per_class()
        return ex_per_class

    def _select_sending_strategy(self):
        def send_a_copy(client: Client, model: Model):
            client.model = copy.deepcopy(model)

        def copy_reference(client: Client, model: Model):
            client.model = model

        if self.__average_model:
            return send_a_copy
        return copy_reference

    def __select_model_computing(self):
        if self.__average_model:
            return self.__compute_averaged
        return self.__take_last_trained

    def __clients_dropping(self):
        assert 0 <= self.__clients_dropout < 1, f"Dropout rate d must be 0 <= d < 1, got {self.__clients_dropout}"
        if self.__clients_dropout > 0:
            return lambda x: select_random_subset(x, self.__clients_dropout)
        return lambda x: x
