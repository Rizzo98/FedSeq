from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import copy
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from src.algo.fed_clients.base_client import Client
import logging
from tqdm import tqdm
import src.utils.aws_cv_task2vec.models as models
import src.utils.aws_cv_task2vec.task2vec as t2v
from src.utils.data import dataset_from_dataloader
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ClientEvaluation:
    def __init__(self, representers: List[np.ndarray], extracted: str):
        self.__extracted = extracted
        self.__representers = representers

    @property
    def extracted(self):
        return self.__extracted

    @property
    def representers(self):
        return self.__representers


class ClientEvaluator:
    can_extract = ["confidence", "classifierLast", "classifierLast2", "classifierAll", "task2vec", "classDistribution"]

    def __init__(self, exemplar_dataset, model, extract: List[str], variance_explained: float, epochs: int,
                device: str, task2vec: dict, *args, **kwargs):
        known_extraction = all(to_extract in ClientEvaluator.can_extract for to_extract in extract)
        assert known_extraction, "Unknown method to evaluate clients"
        assert 0 <= variance_explained <= 1, f"Illegal value, expected 0 <= variance_explained <= 1, given {variance_explained}"
        self.exemplar_dataset = exemplar_dataset
        self.exemplar_dataloader = DataLoader(exemplar_dataset, num_workers=0, batch_size=1)
        self.model = model if all(to_extract != 'task2vec' for to_extract in extract) else models.get_model(task2vec.probe_network, pretrained=True, num_classes=model.num_classes).to(device=device)
        self.extract = extract
        self.variance_explained = variance_explained
        self.epochs = epochs
        if any(to_extract == 'task2vec' for to_extract in extract):
            self.task2vec_method = task2vec.method
            self.task2vec_pn = task2vec.probe_network
            
        

    def evaluate(self, clients: List[Client], optimizer, optimizer_args, loss_class, save_representers = False) -> Dict[str, ClientEvaluation]:
        evaluations = {}
        representers = {e: list() for e in self.extract}
        for client in tqdm(clients, desc='Pretraining of clients'):
            self.__client_pre_train(client, optimizer, optimizer_args, loss_class, self.extract)
            for to_extract in self.extract:
                client_representer = self.__get_representer(client, to_extract, save_representers)
                representers[to_extract].append(client_representer)
            client.model = None  # to save space in GPU
        for to_extract in self.extract:
            reduced = self.__reduce_representers(representers[to_extract], to_extract)
            evaluations[to_extract] = ClientEvaluation(reduced, to_extract)
        return evaluations

    def __client_pre_train(self, client: Client, optimizer, optimizer_args, loss_class, extract):
        if all(((to_extract != 'task2vec' and to_extract != 'classDistribution') or (to_extract == 'task2vec' and self.task2vec_pn == 'rnn')) for to_extract in extract):
            loss_fn = loss_class()
            old_dataloader = client.dataloader
            new_dataloader = DataLoader(old_dataloader.dataset, old_dataloader.batch_size, True,
                                        drop_last=self.model.has_batchnorm())
            client.dataloader = new_dataloader
            self.__send_model(client)
            client.client_update(optimizer, optimizer_args, self.epochs, loss_fn)
            client.dataloader = old_dataloader

    def __get_representer(self, client: Client, to_extract: str, save_representers:bool) -> np.ndarray:
        if to_extract == "confidence":
            return self.__get_prediction(client, save_representers)
        elif to_extract == "task2vec":
            if self.task2vec_pn != 'rnn':
                self.task2vec = t2v.Task2Vec(copy.deepcopy(self.model), skip_layers=self.model.skip_layers, method=self.task2vec_method, classifier_opts=self.model.classifier_opts, loader_opts = self.model.loader_opts)
            else:
                self.task2vec = t2v.Task2Vec(client.model, max_samples=None, method=self.task2vec_method, classifier_opts={'epochs':0})
            return self.task2vec.embed(dataset_from_dataloader(client.dataloader,self.model.in_channels, self.model.extend_labels)).hessian
        elif to_extract == "classDistribution":
            return self.__get_class_distribution(client.dataloader.dataset)
        else:
            fc_layers = ClientEvaluator.extract_fully_connected(client.model)
            if to_extract == "classifierLast":
                return fc_layers[-1]
            elif to_extract == "classifierLast2":
                return np.concatenate([fc_layers[-1], fc_layers[-2]])
            else:
                return np.concatenate(fc_layers)

    def __reduce_representers(self, representers: List[np.ndarray], to_extract: str):
        if self.variance_explained > 0 and ((to_extract != "confidence" and to_extract != "task2vec" and to_extract != "classDistribution") or self.task2vec_pn == 'minGPT'):
            n_components_before = len(representers[0])
            if len(representers)*n_components_before<5_000*1_000_000:
                scaler = StandardScaler()
                representers = scaler.fit_transform(representers)
                reducer = PCA(n_components=self.variance_explained, svd_solver='full')
                new_representers = reducer.fit_transform(representers)
                log.info(
                f"PCA with var_expl={self.variance_explained} on {to_extract}, kept {reducer.n_components_}/{n_components_before} components")
            else: #too much ram!
                new_representers = None
                chunk_size=500
                reducer = IncrementalPCA(n_components=15)
                for i in range(0,len(representers),chunk_size):
                    max_index = min(len(representers),i+chunk_size)
                    reducer.partial_fit(representers[i:max_index])
                for i in range(0,len(representers),chunk_size):
                    max_index = min(len(representers),i+chunk_size)
                    if i==0:
                        new_representers = reducer.transform(representers[i:max_index])
                    else:
                        new_representers = np.vstack((new_representers,reducer.transform(representers[i:max_index])))
                log.info(f'Incremental PCA used from {n_components_before} components to 15.')
            return new_representers
        return representers

    def __send_model(self, client: Client):
        client.model = copy.deepcopy(self.model)

    def __get_model(self, client: Client):
        return client.model

    @staticmethod
    def extract_fully_connected(model) -> List[np.ndarray]:
        fc_layers = []
        for _, layer in model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                fc_layers.append(layer.weight.detach().cpu().numpy().flatten())
        return fc_layers

    def __get_prediction(self, client: Client, save_representers:bool):
        model = self.__get_model(client)
        if not save_representers:
            model.to("cpu")
        model.eval()
        n_classes = self.model.num_classes
        conf_vector = np.zeros(n_classes)
        with torch.no_grad():
            for exemplar, target in self.exemplar_dataloader:
                if save_representers:
                    exemplar = exemplar.to('cuda:0')
                    target = target.to('cuda:0')
                logits = model(exemplar)[0].detach().cpu().numpy()
                logits = softmax(logits)
                conf_vector[target] += logits[target]
                
        conf_vector = conf_vector / np.sum(conf_vector)
        return conf_vector
    
    def __get_class_distribution(self, dataset: Dataset):
        num_classes = dataset.num_classes
        labels = np.array(dataset.labels)
        return np.bincount(labels, minlength=num_classes)/len(labels)
        

