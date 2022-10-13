from src.algo import FedSeq, ASAM
import numpy as np
import copy
from tqdm import tqdm
class FedSeqASAM(FedSeq, ASAM):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf=wandbConf)
        self.swa_start = None
        self.c = params.c
        self.lr2 = params.lr2
        
    def train_step(self):
        if self.swa_start == None:
            self.swa_start = int(.75 * self.num_round)
        n_sample = max(int(self.fraction * self.num_superclients), 1)
        sample_set = np.random.choice(range(self.num_superclients), n_sample, replace=False)
        self.selected_clients = [self.superclients[k] for k in iter(sample_set)]
        self.send_data(self.selected_clients)
        if self._round == self.swa_start:
            self.center_server.swa_model = copy.deepcopy(self.center_server.model)
        current_optim_args = self.optimizer_args
        if self._round >= self.swa_start and self.c > 1:
            t = 1 / self.c * (self._round % self.c + 1)
            lr = (1 - t) * self.optimizer_args['lr'] + t * self.lr2
            current_optim_args['lr'] = lr
            
        for c in tqdm(self.selected_clients, desc=f'Training of selected superclients @ round {self._round}'):
            c.client_update(self.optimizer, current_optim_args, self.training.sequential_rounds, self.loss_fn)