from math import floor
from src.utils.utils import WanDBSummaryWriter
from src.algo import FedAvg
from src.algo.fed_clients import FedAvgClient
import numpy as np
import logging
from typing import List, Optional
from torch.utils.data import DataLoader
from src.datasets.cifar import CifarLocalDataset

log = logging.getLogger(__name__)

class Attacker(FedAvgClient):
    def __init__(self, client_id: int, scrambled_classes : List, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.scrambled_classes = scrambled_classes
        self.__flip_labels()

    def __flip_labels(self):
        dataset : CifarLocalDataset = self.dataloader.dataset
        self.total_flips = [0 for _ in range(len(self.scrambled_classes))]
        for i in range(len(dataset)):
            for j,(s_class, t_class) in enumerate(self.scrambled_classes):
                if dataset.labels[i] == s_class:
                    dataset.labels[i] = t_class
                elif dataset.labels[i] == t_class:
                    dataset.labels[i] = s_class
                self.total_flips[j]+=1


class FedAvgLabelFlippingAttack(FedAvg):
    def __init__(self, model_info, params, device: str, dataset: str, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        self.percentage_client_infected = params['percentage_client_infected']
        self.scramble_method = params['scramble_method']
        self.scrambled_classes = params['scrambled_classes']
        
        assert self.scramble_method in ['random','fixed'], 'Wrong scramble method!'
        assert len(self.scrambled_classes)>1 and len(self.scrambled_classes)<=self.dataset_num_classes, 'Wrong number of classes'
        assert len(set(self.scrambled_classes)) == len(self.scrambled_classes), 'At least one class is repeated'
        assert self.percentage_client_infected>=0 and self.percentage_client_infected<=1,\
            'Wrong value for percentage_client_infected'
        
        clients_ids = self.__injectAttacker()
        log.info(f'Clients infested : {clients_ids}')

    def __createScramblePairs(self):
        shifted_list = list(np.roll(np.array(self.scrambled_classes),np.random.randint(1,len(self.scrambled_classes))))
        return [(self.scrambled_classes[i],shifted_list[i]) for i in range(len(self.scrambled_classes))]

    def __injectAttacker(self):
        number_clients = max(1,floor(len(self.clients)*self.percentage_client_infected))
        clients_pos : List[FedAvgClient] = np.random.choice(list(range(len(self.clients))), number_clients,replace=False)
        clients_ids = []

        scramble_table = [[] for _ in range(number_clients)]
        flips_table = [[] for _ in range(number_clients)]

        if self.scramble_method == 'fixed': scrambled_classes = self.__createScramblePairs() 

        for i,p in enumerate(clients_pos):
            client : FedAvgClient = self.clients[p]
            if self.scramble_method == 'random': scrambled_classes = self.__createScramblePairs()
            attacker = Attacker(client.client_id, scrambled_classes, client.dataloader, client.num_classes, client.device, client.dp)
            scramble_table[i] += [attacker.client_id]
            scramble_table[i] += [f'{c1}-{c2}' for c1,c2 in scrambled_classes]
            flips_table[i] += [attacker.client_id]
            flips_table[i] += attacker.total_flips
            clients_ids.append(client.client_id)
            self.clients[p] = attacker

        columns = ['Attacker_id']+[f'Scramble {i}' for i in range(len(self.scrambled_classes))]
        self.writer.add_table(scramble_table,columns,'Scrambling classes')

        columns = ['Attacker_id']+[f'# of scrambling {i}' for i in range(len(self.scrambled_classes))]
        self.writer.add_table(flips_table,columns,'# of scrambling')

        return clients_ids
    
    def train_step(self):
        super().train_step()
        writer : WanDBSummaryWriter =  self.writer
        for c in self.scrambled_classes:
            acc = self.result['accuracy_class'][-1][c].item()
            writer.add_scalar(f'Acc class {c}', acc, self._round)
