import json
from src.algo import FedSeq
from src.algo.fed_clients import FedAvgClient
from src.algo.fedseq_modules.superclient import FedSeqSuperClient
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt
import os
from src.datasets.cifar import CifarLocalDataset
from src.models.GAN_models import Cifar10Generator, Cifar10Discriminator, EmnistGenerator, EmnistDiscriminator

log = logging.getLogger(__name__)

class Attacker(FedAvgClient):
    def __init__(self, client_id: int, dataloader: Optional[DataLoader], G_class, D_class, outputFolder, saveImages, saveLoss, saveDataset, num_classes=None, device="cpu", dp=None):
        super().__init__(client_id, dataloader, num_classes, device, dp)
        self.__G_class = G_class
        self.__D_class = D_class
        self.__saveDataset = saveDataset
        self.G = self.__G_class()
        self.G.weight_init(mean=0.0, std=0.02)
        self.train_epochs = 300
        self.class_attacked = 3
        self.b_size = 64
        self.__attack_dataloader = DataLoader(self.dataloader.dataset,batch_size=self.b_size,shuffle=True)
        self.n_generation = 16
        self.__fixed_noise = Variable(torch.randn(self.n_generation,100,1,1).to(self.device))
        self.fake_label = 5
        self.__round = -1
        self.__accuracy = -1
        self.__accuracy_class = -1
        self.__original_data = None
        self.__all_data = None
        self.__attack_folder = outputFolder
        self.__save_images = saveImages
        self.__save_loss = saveLoss
        if not os.path.isdir(self.__attack_folder):
            os.mkdir(self.__attack_folder)
    
    @property
    def round(self):
        return self.__round
    
    @round.setter
    def round(self, r):
        self.__round = r

    @property
    def discriminator_accuracy(self):
        return self.__accuracy
    
    @discriminator_accuracy.setter
    def discriminator_accuracy(self, a):
        self.__accuracy = a[-1]
    
    @property
    def discriminator_accuracy_class_attacked(self):
        return self.__accuracy_class
    
    @discriminator_accuracy_class_attacked.setter
    def discriminator_accuracy_class_attacked(self, ac):
        self.__accuracy_class = ac[-1][self.class_attacked]

    @property
    def original_data(self):
        return self.__original_data

    @original_data.setter
    def original_data(self, d):
        self.__original_data = d
    
    @property
    def all_data(self):
        return self.__all_data
    
    @all_data.setter
    def all_data(self, d:list):
        self.__all_data = d

    def __add_images_to_dataset(self,images):
        augmented_dataset : CifarLocalDataset = self.dataloader.dataset
        i = images.detach().cpu().numpy()
        i = np.swapaxes(i,1,3)
        i = np.swapaxes(i,1,2)
        augmented_dataset.images=np.concatenate([augmented_dataset.images,i])
        labels = np.array([self.fake_label]*len(images))
        augmented_dataset.labels=np.concatenate([augmented_dataset.labels,labels])
        self.dataloader = DataLoader(augmented_dataset,self.b_size,shuffle=True)
    
    def __get_most_similar(self, img: torch.Tensor):
        img_flat = img.detach().cpu().flatten()
        best = -1
        best_sim = 0
        for i,(original,_) in enumerate(self.original_data):
            sim = nn.CosineSimilarity(dim=0)(img_flat,original.flatten()) 
            if sim >= best_sim:
                best_sim = sim
                best = i

        best_all = (-1,-1)
        best_sim_all = 0
        client_most_similar = -1
        for c_id,client in tqdm(enumerate(self.all_data),desc='Exploring clients'):
            for i,(original,_) in enumerate(client):
                sim = nn.CosineSimilarity(dim=0)(img_flat,original.flatten())
                if sim >= best_sim_all:
                    best_sim_all = sim
                    best_all = (c_id,i)
                    client_most_similar = c_id

        return self.original_data[best][0], self.all_data[best_all[0]][best_all[1]][0],\
            best_sim.item(), best_sim_all.item(), client_most_similar

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.G.to(self.device)
        self.D = self.__D_class(self.model)
        self.D.to(self.device)
        BCE_loss = nn.BCELoss()
        G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        D_loss_real_storage = []
        D_loss_fake_storage = []
        G_loss_storage = []

        for _ in tqdm(range(self.train_epochs),desc='Training GAN'):
            D_loss_real_batch = []
            D_loss_fake_batch = []
            G_loss_batch = []
            for real_images,_ in self.__attack_dataloader:
                real_images = real_images.to(self.device)
                self.D.zero_grad()
                real_D_logits = self.D(real_images)
                D_loss_real = BCE_loss(real_D_logits,torch.normal(mean=1.0,std=0.1,size=(real_images.shape[0],),device=self.device))
                D_loss_real.backward()
                D_loss_real_batch.append(D_loss_real.item())

                z_ = torch.randn(self.b_size, 100, 1, 1)
                z_ = Variable(z_.to(self.device))
                fake_images = self.G(z_)
                fake_D_logits = self.D(fake_images.detach())
                D_loss_fake = BCE_loss(fake_D_logits, torch.normal(mean=0.0,std=0.1,size=(fake_images.shape[0],),device=self.device))
                D_loss_fake.backward()
                D_loss_fake_batch.append(D_loss_fake.item())
                D_optimizer.step()

                self.G.zero_grad()
                output = self.D(fake_images)
                G_loss = BCE_loss(output, torch.ones(self.b_size,device=self.device))
                G_loss.backward()
                G_loss_batch.append(G_loss.item())
                G_optimizer.step()
            D_loss_real_storage.append(np.mean(D_loss_real_batch))
            D_loss_fake_storage.append(np.mean(D_loss_fake_batch))
            G_loss_storage.append(np.mean(G_loss_batch))

        os.mkdir(f'{self.__attack_folder}/round_{self.round}')
        reconstructed_images = self.G(self.__fixed_noise)
        
        if self.__save_loss:
            os.mkdir(f'{self.__attack_folder}/round_{self.round}/Losses')
            plt.plot(list(range(len(D_loss_real_storage))),D_loss_real_storage,label='Real')
            plt.plot(list(range(len(D_loss_fake_storage))),D_loss_fake_storage,label='Fake')
            plt.legend()
            plt.grid()
            plt.title(f'Discriminator loss (acc={self.discriminator_accuracy})')
            plt.savefig(f'{self.__attack_folder}/round_{self.round}/Losses/Discriminator.png')
            plt.close()
            
            plt.plot(list(range(len(G_loss_storage))),G_loss_storage)
            plt.title('Generator loss')
            plt.grid()
            plt.savefig(f'{self.__attack_folder}/round_{self.round}/Losses/Generator.png')
            plt.close()

        if self.__saveDataset:
            os.mkdir(f'{self.__attack_folder}/round_{self.round}/dataset')
            for i,img in enumerate(reconstructed_images):
                save_image(img.detach().cpu(),f'{self.__attack_folder}/round_{self.round}/dataset/generated_image{i}.png',normalize=True)

        if self.__save_images:
            for i,img in enumerate(reconstructed_images): 
                similar_prev, similar_all, best_sim, best_sim_all, client_most_similar = self.__get_most_similar(img) 
                grid = make_grid(torch.stack((img.detach().cpu(),similar_prev,similar_all)))
                save_image(grid,f'{self.__attack_folder}/round_{self.round}/generated_image{i}.png',normalize=True)

        #self.__add_images_to_dataset(reconstructed_images) 

        super().client_update(optimizer, optimizer_args, local_epoch, loss_fn)


class FedSeqGANattack(FedSeq):
    def __init__(self, model_info, params, device: str, dataset, output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super().__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf)
        G_class, D_class = self.__get_GAN_classes()
        self.__outputFolder = params['output_folder']
        self.__saveImages = params['save_images']
        self.__saveLoss = params['save_loss']
        self.__saveDataset = params['save_dataset_explicit']
        self.__save_dataset_per_client = params['save_dataset_per_client']
        self.__save_client_selection = params['save_client_selection']
        self.__save_client_distribution = params['save_client_distribution']
        self.__save_superclient_composition = params['save_superclient_composition']
        self.__save_attacker = params['save_attacker']
        sc, c, pos = self.__injectAttacker(G_class, D_class, self.__outputFolder, self.__saveImages, self.__saveLoss, self.__saveDataset)
        
        if self.__save_attacker:
            json.dump({'Attacker_id':c.client_id, 'Superclient_id':sc.client_id ,'Position':int(pos)},open(f'{self.__outputFolder}/Attacker.json','w'))
        
        if self.__save_superclient_composition:
            sc_comp = dict()
            for s in self.superclients:
                sc_comp[s.client_id] = [c.client_id for c in s.clients]
            json.dump(sc_comp,open(f'{self.__outputFolder}/SuperclientsComposition.json','w'))

        self.attacker.original_data = sc.clients[pos-1].dataloader.dataset

        if self.__save_client_selection: self.client_selection = dict()
        if self.__save_dataset_per_client: os.mkdir(f'{self.__outputFolder}/datasetsPerClient')
        if self.__save_client_distribution: clients_distribution = dict()

        all_data = []
        for s in self.superclients:
            for c in s.clients:
                all_data += c.dataloader.dataset
                if self.__save_client_distribution: clients_distribution[c.client_id] = list(c.num_ex_per_class())
                if self.__save_dataset_per_client: 
                    os.mkdir(f'{self.__outputFolder}/datasetsPerClient/Client{c.client_id}')
                    for i,(img,_) in enumerate(c.dataloader.dataset): 
                        save_image(img,f'{self.__outputFolder}/datasetsPerClient/Client{c.client_id}/Image{i}.png',normalize=True)
        self.attacker.all_data = all_data
        if self.__save_client_distribution: json.dump(clients_distribution,open(f'{self.__outputFolder}/clients_distribution.json','w'))
        log.info(f'Injected attacker on superclient {sc.client_id} at pos {pos}, client id: {c.client_id}')

    def __injectAttacker(self, G_class, D_class, outputFolder, saveImages, saveLoss, saveDataset):
        superclient : FedSeqSuperClient = np.random.choice(self.superclients,1)[0]
        # superclient : FedSeqSuperClient = self.superclients[0]
        pos = np.random.choice(range(1,len(superclient.clients)),1)[0]
        client = superclient.clients[pos]
        self.attacker = Attacker(client.client_id,client.dataloader,G_class,D_class,outputFolder,saveImages,saveLoss,saveDataset,client.num_classes,client.device,client.dp)
        superclient.clients = (pos, self.attacker)
        return superclient, client, pos
    
    def __get_GAN_classes(self):
        if self.dataset.name=='cifar10':
            return Cifar10Generator, Cifar10Discriminator
        if self.dataset.name=='emnist_niid' or self.dataset.name=='emnist_iid':
            return EmnistGenerator, EmnistDiscriminator

    def train_step(self):
        self.attacker.round = self._round
        self.attacker.discriminator_accuracy = self.result['accuracy']
        self.attacker.discriminator_accuracy_class_attacked = self.result['accuracy_class']
        
        # n_sample = max(int(self.fraction * self.num_superclients), 1)
        # sample_set = np.random.choice(range(1,self.num_superclients), n_sample-1, replace=False)
        # self.selected_clients = [self.superclients[0]]+[self.superclients[k] for k in iter(sample_set)]
        # self.send_data(self.selected_clients)

        # for c in tqdm(self.selected_clients, desc=f'Training of selected superclients @ round {self._round}'):
        #     c.client_update(self.optimizer, self.optimizer_args, self.training.sequential_rounds, self.loss_fn)

        # if self.training.check_forgetting:
        #     round_fg_stats = {}
        #     for c in self.selected_clients:
        #         round_fg_stats[c.client_id] = c.forgetting_stats
        #     self.result["forgetting_stats"].append(round_fg_stats)

        super().train_step()
        if self.__save_client_selection:
            self.client_selection[self._round] = []
            for s in self.selected_clients:
                self.client_selection[self._round]+=[c.client_id for c in s.clients]
            json.dump(self.client_selection,open(f'{self.__outputFolder}/client_selection.json','w'))