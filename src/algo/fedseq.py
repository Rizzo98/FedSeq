import logging
import pickle
from typing import List, Dict
from tqdm import tqdm
from src.optim import *
from src.algo.fedseq_modules import *
from src.models import Model
from src.utils import savepickle
import numpy as np
from torch.nn import CrossEntropyLoss
from src.algo import FedBase
from src.algo.center_server import LayerPermCenterServer
import itertools as it

log = logging.getLogger(__name__)


class FedSeq(FedBase):
    def __init__(self, model_info, params, device: str, dataset,
                 output_suffix: str, savedir: str, writer=None, wandbConf=None):
        super(FedSeq, self).__init__(model_info, params, device, dataset, output_suffix, savedir, writer, wandbConf=wandbConf)

        self.clustering = params.clustering
        self.evaluator = params.evaluator
        self.training = params.training
        self.wandbConf = wandbConf
        self.savedir = savedir
        self.save_representers = params.save_representers

        # list incompatibilities between extract method and clustering measures
        self.extractions = {*self.evaluator.extract_eval, self.evaluator.extract}
        self.clustering_measures = {*self.clustering.measures_eval, self.clustering.measure}
        self.incompatibilities = {to_extract: to_extract not in self.evaluator.extract_prop_distr
                                  and self.clustering.disomogeneity_measures or []
                                  for to_extract in self.extractions
                                  }

        # list all clustering methods, the one used later for training and the ones for evaluation
        clustering_methods: Dict[str, ClusterMaker] = {
            m: eval(m)(**self.clustering, num_classes=self.dataset_num_classes,
                       savedir=savedir) for m in
            {*self.clustering.classnames_eval, self.clustering.classname}}

        if params.clustering.precomputed == None:
            if params.evaluator.precomputed == None:
                evaluations = self.evaluate_if_needed(clustering_methods)
            else:
                evaluations = pickle.load(open(params.evaluator.precomputed,'rb'))
                
            if len(self.clustering.classnames_eval) > 0:
                self.run_clustering_evaluation(clustering_methods, evaluations)

            clients_representer = []
            if self.evaluator.extract in evaluations:
                clients_representer = evaluations[self.evaluator.extract].representers

            self.superclients: List[FedSeqSuperClient] = self._run_clustering_training(clustering_methods,clients_representer,
                                                                              self.evaluator.extract)
        else:
            self.superclients: List[FedSeqSuperClient] = pickle.load(open(params.clustering.precomputed,'rb')) 
        
        if params.save_superclients:
            savepickle(self.superclients,f'{savedir}/superclients.pkl')

        if self.wandbConf.superclient_datasets:
            superclients_distrib = [list(s.num_ex_per_class()) for s in self.superclients]
            self.writer.add_table(superclients_distrib,
                [f'Class {j}' for j in range(len(superclients_distrib[0]))],'Superclients distribution')
            superclients_ids = [[c.client_id for c in s.clients] for s in self.superclients]
            max_ids = max([len(s) for s in superclients_ids])
            superclients_ids = [s+[-1]*(max_ids-len(s)) for s in superclients_ids]
            self.writer.add_table(superclients_ids,[f'Client #{i}' for i in range(max_ids)],'Superclients ids')
        
        self.num_superclients = len(self.superclients)
        self.result.update({"forgetting_stats": [{}]})

    def reset_result(self):
        super().reset_result()
        self.result.update({"forgetting_stats": [{}]})

    def evaluate_if_needed(self, clustering_methods) -> Dict[str, ClientEvaluation]:
        for method in clustering_methods.values():
            if method.requires_incompatibility_check():      
                at_least_one_extraction_needed = any(measure not in self.incompatibilities[extr]
                                                    for extr in self.extractions for measure in self.clustering_measures)
                assert at_least_one_extraction_needed, \
                    f"Incompatibility between extraction set={self.extractions} " \
                    f"and clustering measure set={self.clustering.measures_eval}"

        # use the elements extracted from test set as examplars
        exemplar_dataset = self.excluded_from_test

        if type(self.center_server) is LayerPermCenterServer:
            self.center_server.set_exemplar_set(exemplar_dataset)

        evaluations = {}
        if any(m.requires_clients_evaluation() for m in clustering_methods.values()):
            log.info("Evaluating clients")
            model_evaluator = Model(self.model_info, self.dataset_num_classes)
            client_evaluator = ClientEvaluator(exemplar_dataset, model_evaluator, list(self.extractions),
                                               self.evaluator.variance_explained, self.evaluator.epochs)
            optim_class, optim_args = eval(self.evaluator.optim.classname), self.evaluator.optim.args
            evaluations = client_evaluator.evaluate(self.clients, optim_class, optim_args, CrossEntropyLoss,
                save_representers=self.save_representers)
                
        if self.save_representers:
            savepickle(evaluations,f'{self.savedir}/representers.pkl')
        return evaluations

    def run_clustering_evaluation(self, clustering_methods, evaluations: Dict[str, ClientEvaluation]) -> None:
        # mapping extracted representation -> allowed measures on it
        clustering_measures = {
            e.extracted: {*self.clustering.measures_eval}.difference(self.incompatibilities[e.extracted])
            for e in evaluations.values()
        }
        # check that there is at least one measure tha can be used in clustering given the extracted representations
        if all(len(m) == 0 for m in clustering_measures.values()):
            log.warning("No valid combination for clustering algorithms evaluation")
            return

        # for all the extracted features, run clustering
        for e in evaluations.values():
            # pair method, measure for clustering methods in eval that require evaluation and do not use custom metric
            clustering_eval_comb = [((name, method), measure) for (name, method), measure in
                                    it.product(clustering_methods.items(), clustering_measures[e.extracted])
                                    if name in self.clustering.classnames_eval and method.requires_clients_evaluation()
                                    and not (name == self.clustering.classname and measure == self.clustering.measure)
                                    and not method.uses_custom_metric()
                                    ]
            # add for those clustering methods in eval that use a custom metric
            clustering_eval_comb.extend([((name, method), None) for name, method in clustering_methods.items()
                                         if method.uses_custom_metric()])
            log.info(f"From evaluation extracted {e.extracted}")
            self._run_clustering_combos(clustering_eval_comb, e.representers, e.extracted)

        # run clustering for those methods that do not require evaluation
        clustering_eval_comb = [((name, method), None) for name, method in clustering_methods.items()
                                if not method.requires_clients_evaluation() and not name == self.clustering.classname]
        self._run_clustering_combos(clustering_eval_comb, [])

    def _run_clustering_combos(self, clustering_eval_comb, representers: List[np.ndarray], extracted: str = ""):
        for (classname, method), measure in clustering_eval_comb:
            method.save_statistics = True
            method.measure = measure
            log.info(f"Clustering with {classname} using {measure} for clustering evaluation")
            method.make_superclients(self.clients, representers, sub_path=extracted, **self.training)

    def _run_clustering_training(self, clustering_methods, representers: List[np.ndarray], extracted: str = ""):
        method = clustering_methods[self.clustering.classname]
        method.save_statistic = self.clustering.save_statistics
        method.measure = method.requires_clients_evaluation() and self.clustering.measure or None
        log.info(f"Clustering with {self.clustering.classname} using {method.measure} for training")
        return method.make_superclients(self.clients, representers, sub_path=extracted, **self.training,
                                        optimizer_class=self.optimizer, optimizer_args=self.optimizer_args)

    def train_step(self):
        n_sample = max(int(self.fraction * self.num_superclients), 1)
        sample_set = np.random.choice(range(self.num_superclients), n_sample, replace=False)
        self.selected_clients = [self.superclients[k] for k in iter(sample_set)]
        self.send_data(self.selected_clients)

        for c in tqdm(self.selected_clients, desc=f'Training of selected superclients @ round {self._round}'):
            c.client_update(self.optimizer, self.optimizer_args, self.training.sequential_rounds, self.loss_fn)

        if self.training.check_forgetting:
            round_fg_stats = {}
            for c in self.selected_clients:
                round_fg_stats[c.client_id] = c.forgetting_stats
            self.result["forgetting_stats"].append(round_fg_stats)
