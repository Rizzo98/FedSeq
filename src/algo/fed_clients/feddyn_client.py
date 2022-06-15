import copy
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss, functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad

from src.algo.fed_clients.base_client import Client


class FedDynClient(Client):

    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None,
                 alpha=0):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.alpha = alpha
        self.prev_grads = None

    def init_grads(self):
        if self.prev_grads is None:
            self.prev_grads = [torch.zeros_like(p.data, device=self.device) for p in
                               self.model.parameters() if p.requires_grad]

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.model.to(self.device)
        loss_fn.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        self.init_grads()
        prev_model = copy.deepcopy(self.model)

        for _ in range(local_epoch):
            self.model.train()
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)

                linear_p = 0
                for param, grad in zip(self.model.parameters(), self.prev_grads):
                    linear_p += torch.sum(param.data * grad.data)

                quadratic_p = 0
                for cur_param, prev_param in zip(self.model.parameters(), prev_model.parameters()):
                    quadratic_p += F.mse_loss(cur_param, prev_param, reduction='sum')

                loss -= linear_p
                loss += self.alpha / 2. * quadratic_p
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),30)
                optimizer.step()

                for prev_grads, new_params, prev_params in zip(self.prev_grads, self.model.parameters(),
                                                               prev_model.parameters()):
                    prev_grads.add_(new_params.data - prev_params.data, alpha=-self.alpha)
