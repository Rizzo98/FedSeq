from src.algo import FedSeq
from tqdm import tqdm
from src.algo.center_server import ModelsAllignmentCenterServer
import numpy as np

class FedSeqLossTracking(FedSeq):
    
    def train_step(self):
        n_sample = max(int(self.fraction * self.num_superclients), 1)
        sample_set = np.random.choice(range(self.num_superclients), n_sample, replace=False)
        self.selected_clients = [self.superclients[k] for k in iter(sample_set)]
        self.send_data(self.selected_clients)

        for c in tqdm(self.selected_clients, desc=f'Training of selected superclients @ round {self._round}'):
            c.client_update(self.optimizer, self.optimizer_args, self.training.sequential_rounds, self.loss_fn)

        loss_per_sc = dict()
        for sc in tqdm(self.selected_clients, desc='Traking loss of selected superclients'):
            tot_loss = 0
            count = 0
            for c in sc.clients:
                for img, target in c.dataloader:
                    img = img.to(self.device)
                    target = target.to(self.device)
                    logits = sc.model(img)
                    tot_loss += self.loss_fn(logits, target).item()
                    count += 1
            loss_per_sc[sc.client_id] = tot_loss/count

        assert type(self.center_server) == ModelsAllignmentCenterServer,\
            "Center server must be 'ModelsAllignmentCenterServer'!"    
        
        self.center_server.setLossValues(loss_per_sc)

        if self.training.check_forgetting:
            round_fg_stats = {}
            for c in self.selected_clients:
                round_fg_stats[c.client_id] = c.forgetting_stats
            self.result["forgetting_stats"].append(round_fg_stats)