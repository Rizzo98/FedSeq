from multiprocessing.connection import Client
from src.algo import FedSeq
from src.algo.fed_clients import FedAvgClient
from src.algo.fedseq_modules import superclient
from src.algo.fedseq_modules.superclient import FedSeqSuperClient
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.datasets.cifar import CifarLocalDataset

log = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=32, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            Generator.normal_init(self._modules[m], mean, std)

    def forward(self, input):
        return self.main(input)


class Attacker(FedAvgClient):
    def __init__(self, client_id: int, dataloader: Optional[DataLoader], num_classes=None, device="cpu", dp=None):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.train_epochs = 100
        self.class_attacked = 3
        self.b_size = 16
        self.n_generation = 16
        self.fake_label = 5
        self.__round = -1
    
    @property
    def round(self):
        return self.__round
    
    @round.setter
    def round(self, r):
        self.__round = r

    def __add_images_to_dataset(self,images):
        augmented_dataset : CifarLocalDataset = self.dataloader.dataset
        i = images.detach().numpy()
        i = np.swapaxes(i,1,3)
        i = np.swapaxes(i,1,2)
        augmented_dataset.images=np.concatenate([augmented_dataset.images,i])
        labels = np.array([self.fake_label]*len(images))
        augmented_dataset.labels=np.concatenate([augmented_dataset.labels,labels])
        self.dataloader = DataLoader(augmented_dataset,self.b_size,shuffle=True)

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        G = Generator()
        G.weight_init(mean=0.0, std=0.02)
        G.to(self.device)
        BCE_loss = nn.BCELoss()
        G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        for _ in range(self.train_epochs):
            G.zero_grad()
            z_ = torch.randn(self.b_size, 100, 1, 1)
            z_ = Variable(z_.to(self.device))
            G_result = G(z_)
            D_result = F.softmax(self.model(G_result),dim=1)
            G_train_loss = BCE_loss(D_result[:,self.class_attacked], torch.ones(self.b_size))
            G_train_loss.backward()
            G_optimizer.step()
        
        reconstructed_images = G(torch.randn(self.n_generation,100,1,1))
        self.__add_images_to_dataset(reconstructed_images)
        grid = make_grid(reconstructed_images)
        save_image(grid,f'/home/arizzardi/GAN_attack_images/round_{self.round}.png')
        super().client_update(optimizer, optimizer_args, local_epoch, loss_fn)


class FedSeqGANattack(FedSeq):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        sc_id, c_id = self.__injectAttacker()
        log.info(f'Injected attacker on superclient {sc_id}, client {c_id}')

    def __injectAttacker(self):
        superclient : FedSeqSuperClient = np.random.choice(self.superclients,1)[0]
        pos = np.random.choice(range(len(superclient.clients)),1)[0]
        client = superclient.clients[pos]
        self.attacker = Attacker(client.client_id,client.dataloader,client.num_classes,client.device,client.dp)
        superclient.clients = (pos, self.attacker)
        return superclient.client_id, client.client_id
    
    def train_step(self):
        self.attacker.round = self._round
        super().train_step()