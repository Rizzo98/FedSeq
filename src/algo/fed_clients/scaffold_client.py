import copy
from typing import Optional
import torch
from torch.utils.data import DataLoader
from src.algo.fed_clients.base_client import Client


class SCAFFOLDClient(Client):

    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.old_controls = None
        self.controls = None
        self.server_controls = None

    @staticmethod
    def from_base_client(c: Client):
        return SCAFFOLDClient(c.client_id, c.dataloader, c.num_classes, c.device, c.dp)

    def init_controls(self):
        if self.controls is None:
            self.controls = [torch.zeros_like(p.data, device="cpu") for p in
                             self.model.parameters() if p.requires_grad]

    def move_controls(self, device: str):
        for i in range(len(self.controls)):
            self.controls[i] = self.controls[i].to(device)
            self.server_controls[i] = self.server_controls[i].to(device)

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.init_controls()
        # save model sent by server for computing delta_model
        server_model = copy.deepcopy(self.model)
        self.model.train()
        op = optimizer(self.model.parameters(), **optimizer_args)
        for _ in range(local_epoch):
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                op.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)
                loss.backward()

                # put controls in correct device before passing to optimizer
                self.move_controls(self.device)
                op.step(self.server_controls, self.controls)
                self.move_controls("cpu")

        # controls become old controls
        self.old_controls = [torch.clone(c) for c in self.controls]

        # get new controls option 1 of scaffold algorithm
        batches = 0
        op = optimizer(server_model.parameters(), **optimizer_args)
        for _ in range(local_epoch):
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = server_model(img)
                loss = loss_fn(logits, target)
                loss.backward()
                batches += 1
        server_model.to("cpu")
        for cc, p in zip(self.controls, server_model.parameters()):
            cc.data = p.grad.data/batches

    def delta_controls(self):
        delta = [torch.zeros_like(c, device="cpu") for c in self.controls]
        for d, new, old in zip(delta, self.controls, self.old_controls):
            d.data = new.data - old.data
        return delta
