from src.algo import FedSeq, ASAM
import numpy as np
import copy
from tqdm import tqdm
class FedSeqASAM(FedSeq, ASAM):
    def train_step(self):
        n_sample = max(int(self.fraction * self.num_superclients), 1)
        sample_set = np.random.choice(range(self.num_superclients), n_sample, replace=False)
        self.selected_clients = [self.superclients[k] for k in iter(sample_set)]
        self.send_data(self.selected_clients)
        if self._round == self.swa_start:
            self.center_server.swa_model = copy.deepcopy(self.center_server.client_model)
        current_optim_args = self.optimizer_args
        if self._round >= self.swa_start and self.c > 1:
            t = 1 / self.c * (self._round % self.c + 1)
            lr = (1 - t) * self.optimizer_args['lr'] + t * self.lr2
            current_optim_args['lr'] = lr
            
        for c in tqdm(self.selected_clients, desc=f'Training of selected superclients @ round {self._round}'):
            c.client_update(self.optimizer, current_optim_args, self.training.sequential_rounds, self.loss_fn)