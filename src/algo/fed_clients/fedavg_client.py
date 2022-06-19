import copy
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss, functional as F
from torch.utils.data import DataLoader

from src.algo import Algo
from src.algo.fed_clients.base_client import Client

class FedAvgClient(Client):

    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None):
        super().__init__(client_id, dataloader, num_classes, device, dp)

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        if type(loss_fn) != CrossEntropyLoss:
            return self.client_update_distillation(optimizer, optimizer_args, local_epoch, loss_fn)
        self.model.to(self.device)
        loss_fn.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        for _ in range(local_epoch):
            Algo.train(self.model, self.device, optimizer, loss_fn, self.dataloader)
        self.model.to('cpu')
        
    def client_update_distillation(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.model.to(self.device)
        loss_fn.to(self.device)
        self.model.eval()
        tot_logits = []
        with torch.no_grad():
            for img, _ in self.dataloader:
                img = img.to(self.device)
                tot_logits.append(self.model(img))

        self.model.train()
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        for i in range(local_epoch):
            for j, (img, target) in enumerate(self.dataloader):
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, tot_logits[j], target)
                loss.backward()
                optimizer.step()
