from typing import List
from math import floor
from src.utils.utils import WanDBSummaryWriter
from src.algo import FedSeq
from src.algo.fed_clients import FedAvgClient
from src.algo.fedseq_modules.superclient import FedSeqSuperClient
import numpy as np
import logging
from typing import Optional
from torch.utils.data import DataLoader
from src.datasets.cifar import CifarLocalDataset

log = logging.getLogger(__name__)

class Attacker(FedAvgClient):
    def __init__(self, client_id: int, source_class: int, target_class: int, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.source_class = source_class
        self.target_class = target_class
        assert self.source_class in range(self.num_classes), 'Wrong value for source class'
        assert self.target_class in range(self.num_classes), 'Wrong value for target class'
        self.__flip_labels()

    def __flip_labels(self):
        dataset : CifarLocalDataset = self.dataloader.dataset
        self.total_flips_source_to_target = 0
        self.total_flips_target_to_source = 0
        for i in range(len(dataset)):
            if dataset.labels[i] == self.source_class:
                dataset.labels[i] = self.target_class
                self.total_flips_source_to_target+=1
            elif dataset.labels[i] == self.target_class:
                dataset.labels[i] = self.source_class
                self.total_flips_target_to_source+=1


class FedSeqLabelFlippingAttack(FedSeq):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        self.percentage_superclient_infected = params['percentage_superclient_infected']
        self.percentage_client_infected_in_superclient = params['percentage_client_infected_in_superclient']
        self.source_class = params['source_class']
        self.target_class = params['target_class']

        assert self.percentage_client_infected_in_superclient>=0 and self.percentage_client_infected_in_superclient<=1,\
            'Wrong value for percentage_client_infected_in_superclient'
        assert self.percentage_superclient_infected>=0 and self.percentage_superclient_infected<=1,\
            'Wrong value for percentage_superclient_infected'
        
        superclients_ids, clients_ids, total_flips_source_to_target, total_flips_target_to_source = self.__injectAttacker()
        log.info(f'Superclients infested : {superclients_ids}, attackers ids : {clients_ids}')
        log.info(f'Total flips source to target: {total_flips_source_to_target}, Total flips target to source: {total_flips_target_to_source}')

    def __injectAttacker(self):
        number_superclients = max(1,floor(len(self.superclients)*self.percentage_superclient_infected))
        superclients : List[FedSeqSuperClient] = np.random.choice(self.superclients,number_superclients)
        superclients_ids = [s.client_id for s in superclients]
        clients_ids = []
        total_flips_source_to_target = 0
        total_flips_target_to_source = 0

        for s in superclients:
            number_clients = max(1,floor(len(s.clients)*self.percentage_client_infected_in_superclient))
            positions = np.random.choice(range(len(s.clients)),number_clients)
            for p in positions:
                client : FedAvgClient = s.clients[p]
                attacker = Attacker(client.client_id, self.source_class, self.target_class, client.dataloader, client.num_classes, client.device, client.dp)
                s.clients = (p, attacker)
                clients_ids.append(client.client_id)
                total_flips_source_to_target+=attacker.total_flips_source_to_target
                total_flips_target_to_source+=attacker.total_flips_target_to_source

        return superclients_ids, clients_ids, total_flips_source_to_target, total_flips_target_to_source
    
    def train_step(self):
        super().train_step()
        acc_source_class = self.result['accuracy_class'][-1][self.source_class].item()
        acc_target_class = self.result['accuracy_class'][-1][self.target_class].item()
        writer : WanDBSummaryWriter =  self.writer
        writer.add_scalar('Acc source class', acc_source_class, self._round)
        writer.add_scalar('Acc target class', acc_target_class, self._round)