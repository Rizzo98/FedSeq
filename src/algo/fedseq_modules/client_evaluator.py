from typing import List, Dict
from torch.utils.data import DataLoader
import numpy as np
import torch
import copy
from sklearn.decomposition import PCA
from src.algo.fed_clients.base_client import Client
import logging

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
    can_extract = ["confidence", "classifierLast", "classifierLast2", "classifierAll"]

    def __init__(self, exemplar_dataset, model, extract: List[str], variance_explained: float, epochs: int, *args,
                 **kwargs):
        known_extraction = all(to_extract in ClientEvaluator.can_extract for to_extract in extract)
        assert known_extraction, "Unknown method to evaluate clients"
        assert 0 <= variance_explained <= 1, f"Illegal value, expected 0 <= variance_explained <= 1, given {variance_explained}"
        self.exemplar_dataset = exemplar_dataset
        self.exemplar_dataloader = DataLoader(exemplar_dataset, num_workers=0, batch_size=1)
        self.model = model
        self.extract = extract
        self.variance_explained = variance_explained
        self.epochs = epochs

    def evaluate(self, clients: List[Client], optimizer, optimizer_args, loss_class) -> Dict[str, ClientEvaluation]:
        evaluations = {}
        representers = {e: list() for e in self.extract}
        for client in clients:
            self.__client_pre_train(client, optimizer, optimizer_args, loss_class)
            for to_extract in self.extract:
                client_representer = self.__get_representer(client, to_extract)
                representers[to_extract].append(client_representer)
            client.model = None  # to save space in GPU
        for to_extract in self.extract:
            reduced = self.__reduce_representers(representers[to_extract], to_extract)
            evaluations[to_extract] = ClientEvaluation(reduced, to_extract)
        return evaluations

    def __client_pre_train(self, client: Client, optimizer, optimizer_args, loss_class):
        loss_fn = loss_class()
        old_dataloader = client.dataloader
        new_dataloader = DataLoader(old_dataloader.dataset, old_dataloader.batch_size, True,
                                    drop_last=self.model.has_batchnorm())
        client.dataloader = new_dataloader
        self.__send_model(client)
        client.client_update(optimizer, optimizer_args, self.epochs, loss_fn)
        client.dataloader = old_dataloader

    def __get_representer(self, client: Client, to_extract: str) -> np.ndarray:
        if to_extract == "confidence":
            return self.__get_prediction(client)
        else:
            fc_layers = ClientEvaluator.extract_fully_connected(client.model)
            if to_extract == "classifierLast":
                return fc_layers[-1]
            elif to_extract == "classifierLast2":
                return np.concatenate([fc_layers[-1], fc_layers[-2]])
            else:
                return np.concatenate(fc_layers)

    def __reduce_representers(self, representers: List[np.ndarray], to_extract: str):
        if self.variance_explained > 0 and to_extract != "confidence":
            n_components_before = len(representers[0])
            reducer = PCA(n_components=self.variance_explained, svd_solver='full')
            new_representers = reducer.fit_transform(representers)
            log.info(
                f"PCA with var_expl={self.variance_explained} on {to_extract}, kept {reducer.n_components_}/{n_components_before} components")
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

    def __get_prediction(self, client: Client):
        model = self.__get_model(client)
        model.to("cpu")
        model.eval()
        n_classes = self.model.num_classes
        conf_vector = np.zeros(n_classes)
        with torch.no_grad():
            for exemplar, target in self.exemplar_dataloader:
                logits = model(exemplar)[0].detach().cpu().numpy()
                logits = softmax(logits)
                conf_vector[target] += logits[target]
        conf_vector = conf_vector / np.sum(conf_vector)
        return conf_vector
