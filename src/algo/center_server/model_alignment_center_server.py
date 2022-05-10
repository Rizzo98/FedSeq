from tkinter import FALSE
from matplotlib.pyplot import axis
from src.algo.center_server import FedAvgCenterServer
from typing import List
from src.algo.fed_clients import Client
from src.algo import Algo
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import SGD
import torch

class ModelAlignmentCenterServer(FedAvgCenterServer):
    
    def __generate_random_sample(self, dataset_name:str, num_example:int, clients: List[Client])->DataLoader:
        size = None
        classes  = None
        batch_size = 4

        if dataset_name=='cifar10':
            size = (num_example,3,32,32)
            classes = 10
        elif dataset_name=='emnist':
            pass

        assert size!=None, 'Dataset not correct!'
        data = torch.normal(0,5,size)
        noise_dataset = TensorDataset(data)
        dloader = DataLoader(noise_dataset,batch_size=batch_size)
        avg_y = torch.zeros((num_example,classes),device=self.device)
        for sc in clients:
            for i,img in enumerate(dloader):
                img = img[0].to(self.device)
                y = sc.send_model()(img)
                for j in range(batch_size):
                    avg_y[i*batch_size+j] = torch.add(avg_y[i*batch_size+j],y[j])
        
        avg_y_nograd = torch.zeros((num_example,classes),device=self.device)
        for i in range(num_example):
            avg_y_nograd[i] = (avg_y[i].detach()/len(clients))

        return DataLoader(TensorDataset(data,avg_y_nograd),batch_size=batch_size)

    def __agnostic_loss(self, pred, target):
        mean_batch_pred = torch.mean(pred,axis=0)
        mean_batch_target = torch.mean(target,axis=0)
        return torch.mean(mean_batch_pred*(torch.div(mean_batch_pred,2)-mean_batch_target))

    def aggregation(self, clients: List[Client], aggregation_weights: List[float], round: int):
        noise_data = self.__generate_random_sample('cifar10',128,clients)
        epochs=10
        for sc in clients:
            optim = SGD(sc.send_model().parameters(),lr=0.01)
            for _ in range(epochs):
                Algo.train(sc.send_model(),self.device,optim,self.__agnostic_loss,noise_data)

        FedAvgCenterServer.weighted_aggregation([c.send_model() for c in clients], aggregation_weights, self.model)