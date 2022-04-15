import logging
import numpy as np
from tqdm import tqdm
from src.algo.fedbase import FedBase
from src.utils import select_random_subset

log = logging.getLogger(__name__)


class FedAvg(FedBase):
    def __init__(self, model_info, params, device: str, dataset: str,
                 output_suffix: str, savedir: str, writer=None, wandbConf=None):
        assert 0 <= params.clients_dropout < 1, f"Dropout rate d must be 0 <= d < 1, got {params.clients_dropout}"
        super(FedAvg, self).__init__(model_info, params, device, dataset, output_suffix, savedir, writer,wandbConf=wandbConf)
        self.__clients_dropout = params.clients_dropout
        self.dropping = self.__clients_dropout > 0 and (lambda x: select_random_subset(x, self.__clients_dropout)) \
                        or (lambda x: x)
        self.wandbConf = wandbConf

    def train_step(self):
        n_sample = max(int(self.fraction * self.num_clients), 1)
        sample_set = self.dropping(np.random.choice(range(self.num_clients), n_sample, replace=False))
        self.selected_clients = [self.clients[k] for k in iter(sample_set)]
        self.send_data(self.selected_clients)
        
        for c in tqdm(self.selected_clients, desc=f'Training of selected clients @ round {self._round}'):
            c.client_update(self.optimizer, self.optimizer_args,
                            self.local_epoch, self.loss_fn)
