import numpy as np
from src.algo import FedSeq
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class FedSeqInter(FedSeq):

    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super(FedSeqInter, self).__init__(model_info, params, device, dataset, output_suffix,
             savedir, writer, wandbConf=wandbConf)
        # init model bank
        self.num_clients_train_step = max(int(self.fraction * self.num_superclients), 1)
        self.models_bank = [None] * self.num_clients_train_step
        self.models_num_examples = np.zeros(self.num_clients_train_step)
        aggregation_periods_choices = {"num_superclients": self.num_superclients,
                                       "fraction_superclients": self.num_clients_train_step,
                                       "never": int(1e6)}
        assert params.aggregation_period in aggregation_periods_choices, "Unknown aggregation period"
        self.aggregation_period = aggregation_periods_choices[params.aggregation_period]

    def train_step(self):
        # broadcast aggregated model next round if enough clients
        if self._round % self.aggregation_period == 0:
            log.info("Broadcast aggregated model")
            for k in range(self.num_clients_train_step):
                self.models_bank[k] = self.center_server.send_model()
            self.models_num_examples.fill(0)

        sample_set = np.random.choice(range(self.num_superclients), self.num_clients_train_step, replace=False)
        self.selected_clients = [self.superclients[k] for k in iter(sample_set)]
        
        for k in tqdm(range(self.num_clients_train_step), desc=f'Training of selected superclients @ round {self._round}'):
            # send past round models
            self.selected_clients[k].model = self.models_bank[k]
            # training
            self.selected_clients[k].client_update(self.optimizer, self.optimizer_args,
                                                   self.training.sequential_rounds, self.loss_fn)
            self.models_bank[k] = self.selected_clients[k].send_model()
            self.models_num_examples[k] += len(self.selected_clients[k])

        if self.training.check_forgetting:
            round_fg_stats = {}
            for k in iter(sample_set):
                round_fg_stats[self.superclients[k].client_id] = self.superclients[k].forgetting_stats
            self.result["forgetting_stats"].append(round_fg_stats)

    def aggregate(self, clients):
        total_weight = np.sum(self.models_num_examples)
        weights = [w / total_weight for w in self.models_num_examples]
        self.center_server.aggregation(clients, weights, self._round)
