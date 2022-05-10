from src.algo import FedSeq
from tqdm import tqdm
from src.algo.center_server import LossWeightCenterServer
import numpy as np
import copy
from scipy.special import softmax

class FedSeqLossTracking(FedSeq):

    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        self.prob = [1/self.num_superclients for _ in range(self.num_superclients)]
    
    def train_step(self):
        n_sample = max(int(self.fraction * self.num_superclients), 1)
        sample_set = np.random.choice(range(self.num_superclients), n_sample, replace=False, p=self.prob)
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
        
        '''
        tot_loss = sum(list(loss_per_sc.values()))
        weight_per_sc = dict([(k,tot_loss/v) for k,v in loss_per_sc.items()])
        tot_weight = sum(list(weight_per_sc.values()))
        norm_weight_per_sc = dict([(k,v/tot_weight) for k,v in weight_per_sc.items()])
        prob_copy = copy.deepcopy(self.prob)
        tau=2
        for sc_id, norm_w in norm_weight_per_sc.items():
            to_redistribute = min(prob_copy[sc_id],norm_w*tau)
            single_part = to_redistribute/(len(self.prob)-1)
            self.prob[sc_id]-=to_redistribute
            for k in range(len(self.prob)):    
                if k!=sc_id:
                    self.prob[k]+=single_part
        '''
        penality = 0.5*(1/self.num_superclients)
        mean_loss = np.mean(list(loss_per_sc.values()))
        for sc_id, loss in loss_per_sc.items():
            if loss<mean_loss:
                self.prob[sc_id]=max(self.prob[sc_id]-penality,0)
            else:
                self.prob[sc_id]=min(self.prob[sc_id]+penality,1)
        
        self.prob = [w/sum(self.prob) for w in self.prob]
        

        assert type(self.center_server) == LossWeightCenterServer,\
            "Center server must be 'ModelsAllignmentCenterServer'!"    
        
        self.center_server.setLossValues(loss_per_sc)

        if self.training.check_forgetting:
            round_fg_stats = {}
            for c in self.selected_clients:
                round_fg_stats[c.client_id] = c.forgetting_stats
            self.result["forgetting_stats"].append(round_fg_stats)