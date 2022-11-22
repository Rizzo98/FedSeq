from src.algo.fed_clients.base_client import Client
from typing import Optional
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.optim.asam_opt import ASAM


class ASAMClient(Client):

    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None,
                 rho=0.1, eta=0.2):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.rho = rho
        self.eta = eta

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.model.to(self.device)
        loss_fn.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        for _ in range(local_epoch):
            self.model.train()
            minimizer = ASAM(optimizer, self.model, self.rho, self.eta)
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)

                #Ascent step
                logits = self.model(img)
                loss = loss_fn(logits, target)
                loss.backward()
                minimizer.ascent_step()
                #Descent step
                loss_fn(self.model(img), target).backward()
                minimizer.descent_step()
        self.model.to('cpu')

                