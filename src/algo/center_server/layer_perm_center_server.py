from statistics import mode
from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients.base_client import Client
from src.algo.algo import Algo 
from collections import OrderedDict
from collections import deque
from typing import List
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class LayerPermCenterServer(FedAvgCenterServer):

    def set_exemplar_set(self, exemplar_set):
        batch_size = min(64,len(exemplar_set)//10)
        self.exemplar_loader = DataLoader(exemplar_set, batch_size=batch_size, shuffle=True)

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], round:int):
        models = [c.send_model() for c in clients]
        aggregation_weights = [1/len(models) for _ in range(len(models))]
        layers = list(models[0].state_dict().keys())
        new_models_weights = [OrderedDict() for _ in range(len(models))]
        model_ids = list(range(len(models)))
        t = 1-(round/1000)
        
        np.random.shuffle(model_ids)
        index = max(-1,int(len(model_ids)*t))
        to_permute = model_ids[:index+1]     
        for layer in layers:
            random_shift = np.random.randint(-len(to_permute)-1,+len(to_permute)-1)
            shifted_index = deque(to_permute)
            shifted_index.rotate(random_shift)
            perm_list = zip(to_permute,list(shifted_index))
            for m1, m2 in perm_list:
                new_models_weights[m1][layer] = models[m2].state_dict()[layer]
        [models[i].load_state_dict(new_models_weights[i]) for i in to_permute]

        loss = CrossEntropyLoss()
        for m in tqdm(models,desc='Training perm models'):
            optimizer = SGD(m.parameters(),lr=0.05)
            for _ in range(2):
                Algo.train(m, self.device, optimizer, loss, self.exemplar_loader)
        FedAvgCenterServer.weighted_aggregation([m for m in models], aggregation_weights, self.model)