import random
from typing import List
from math import floor
from src.utils.utils import WanDBSummaryWriter
from src.algo import FedSeq
from src.algo.fed_clients import FedAvgClient, Client
from src.algo.fedseq_modules.superclient import FedSeqSuperClient
from src.algo.fedavg_LabelFlippingAttack import Attacker
import numpy as np
import logging

log = logging.getLogger(__name__)

class FedSeqLabelFlippingAttack(FedSeq):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        self.percentage_superclient_infected = params['percentage_superclient_infected']
        self.percentage_client_infected_in_superclient = params['percentage_client_infected_in_superclient']
        self.scramble_method = params['scramble_method']
        self.scrambled_classes = params['scrambled_classes']
        self.constraints = params['constraints']

        assert self.scramble_method in ['random','fixed'], 'Wrong scramble method!'
        assert len(self.scrambled_classes)>1 and len(self.scrambled_classes)<=self.dataset_num_classes, 'Wrong number of classes'
        assert self.percentage_client_infected_in_superclient>=0 and self.percentage_client_infected_in_superclient<=1,\
            'Wrong value for percentage_client_infected_in_superclient'
        assert self.percentage_superclient_infected>=0 and self.percentage_superclient_infected<=1,\
            'Wrong value for percentage_superclient_infected'
        
        superclients_ids, clients_ids = self.__injectAttacker()
        log.info(f'Superclients infested : {superclients_ids}, attackers ids : {clients_ids}')

    def __createScramblePairs(self):
        assert len(self.scrambled_classes)%2==0, 'Scrambled classes must be even!'
        assert all([len(c)%2==0 for c in self.constraints]), 'Some constraint is not even!'
        scrambled_pairs = []
        for c in self.scrambled_classes:
            if all([c not in t for t in scrambled_pairs]):
                classes_bucket = set(self.scrambled_classes)
                classes_bucket.discard(c)
                for c1,c2 in scrambled_pairs: classes_bucket.discard(c1); classes_bucket.discard(c2)
                constraints = [l for l in self.constraints if c in l]
                if len(constraints)>0:
                    for c_ in self.scrambled_classes:
                        if all([c_ not in cons for cons in constraints]): classes_bucket.discard(c_)
                if len(classes_bucket)==0: raise ValueError('No scrambling possible!')
                chosen = random.choice(tuple(classes_bucket))
                classes_bucket.remove(chosen)
                scrambled_pairs.append((c,chosen))
                if len(scrambled_pairs)==int(len(self.scrambled_classes)/2): break
        return scrambled_pairs

    def __isSuitable(self, c:Client):
        classes = c.num_ex_per_class()
        is_suitable=False
        for cl in self.scrambled_classes:
            if classes[cl]>0: is_suitable=True
        return is_suitable

    def __injectAttacker(self):
        number_superclients = max(1,floor(len(self.superclients)*self.percentage_superclient_infected))
        superclients : List[FedSeqSuperClient] = np.random.choice(self.superclients,number_superclients,replace=False)
        superclients_ids = [s.client_id for s in superclients]
        total_number_clients = sum([max(1,floor(len(s.clients)*self.percentage_client_infected_in_superclient)) for s in superclients])
        clients_ids = []

        scramble_table = [[] for _ in range(total_number_clients)]
        flips_table = [[] for _ in range(total_number_clients)]

        if self.scramble_method == 'fixed': scrambled_classes = self.__createScramblePairs() 

        counter=0
        for s in superclients:
            suitable_clients = [c for c in s.clients if self.__isSuitable(c)]
            number_clients = max(1,floor(len(suitable_clients)*self.percentage_client_infected_in_superclient))
            positions = np.random.choice(range(len(suitable_clients)),number_clients,replace=False)
            for p in positions:
                client : FedAvgClient = s.clients[p]
                if self.scramble_method == 'random': scrambled_classes = self.__createScramblePairs()
                attacker = Attacker(client.client_id, scrambled_classes, client.dataloader, client.num_classes, client.device, client.dp)
                s.clients = (p, attacker)
                clients_ids.append(client.client_id)
                scramble_table[counter]+=[s.client_id]
                scramble_table[counter]+=[client.client_id]
                scramble_table[counter]+=[f'{c1}-{c2}' for c1,c2 in scrambled_classes]
                flips_table[counter]+=[s.client_id]
                flips_table[counter]+=[client.client_id]
                flips_table[counter]+=attacker.total_flips
                counter+=1

        columns = ['Superclient_id','Attacker_id']+[f'Scramble {i}' for i in range(len(self.scrambled_classes)//2)]
        self.writer.add_table(scramble_table,columns,'Scrambling classes')

        columns = ['Superclient_id','Attacker_id']+[f'# of scrambling {i}' for i in range(len(self.scrambled_classes)//2)]
        self.writer.add_table(flips_table,columns,'# of scrambling')

        return superclients_ids, clients_ids
    
    def train_step(self):
        super().train_step()
        writer : WanDBSummaryWriter =  self.writer
        avg = 0
        for c in self.scrambled_classes:
            acc = self.result['accuracy_class'][-1][c].item()
            avg+=acc
            writer.add_scalar(f'Acc class {c}', acc, self._round)
        writer.add_scalar('Avg scrambled classes',avg/len(self.scrambled_classes),self._round)