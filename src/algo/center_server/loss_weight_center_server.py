from src.algo.center_server import FedAvgCenterServer
from src.algo.fed_clients import Client
from typing import Dict, List
import json

class LossWeightCenterServer(FedAvgCenterServer):
    def __init__(self, model, dataloader, device):
        super().__init__(model, dataloader, device)
        self.store_loss = []

    def setLossValues(self, losses : Dict):
        self.__total_loss = sum(list(losses.values()))
        self.__losses = losses

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], round:int):
        d = {'Total_loss': self.__total_loss}
        for c in clients:
            d[f'client_{c.client_id}'] = {'loss':self.__losses[c.client_id],'weight':self.__total_loss/self.__losses[c.client_id]}
        self.store_loss.append(d)
        if round==1000:
            json.dump(self.store_loss,open('/home/arizzardi/lossTrack_round1000_probChange.json','w'))
        
        aggregation_weights = [self.__total_loss/self.__losses[c.client_id] for c in clients]
        aggregation_weights = [a/sum(aggregation_weights) for a in aggregation_weights]
        FedAvgCenterServer.weighted_aggregation([c.send_model() for c in clients], aggregation_weights, self.model)
