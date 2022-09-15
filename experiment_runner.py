from omegaconf import OmegaConf
from src.experiments import *


def main():
    train_defaults = OmegaConf.load('config/config.yaml')
    experiment_config = OmegaConf.load('config/experiments/config.yaml')
    FedExperiment.default_fit_iterations = experiment_config.get("fit_iterations", 1)
    """
    experiments = [
        FedExperiment.from_param_groups("FedAvg - fraction of clients",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedavg"),
                                            Param("common.K", 500),
                                            MultiParam.key("common.C", [0.05, 0.1, 0.2])
                                        ]
                                        ),
        FedExperiment.from_param_groups("FedAvg - K vs alpha",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            MultiParam.key("common.K", [10, 100, 500]),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5, 5, 100]),
                                            Param("common.C", 0.1)
                                        ]
                                        ),
        FedExperiment.from_params("FedSeq - clients pre-training",
                                  "",
                                  Param("algo", "fedseq"),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.dict("algo.params.evaluator", ("epochs", [1, 5, 10, 20, 30, 40])),
                                  MultiParam.dict("algo.params.clustering", ("classname", ["GreedyClusterMaker"])),
                                  Param("do_train", False),
                                  runner_options={"--time": "00:45:00"}
                                  ),
        FedExperiment.from_param_groups("FedAvg - baseline comparison",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedavg"),
                                            Param("common.K", 500),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5, 5, 100]),
                                            Param("common.C", 0.2)
                                        ],
                                        runner_options={"--time": "3-00:00:00"}
                                        ),
        FedExperiment.from_params("FedSeq - clustering comparison lenet",
                                  "",
                                  Param("algo.params.evaluator.extract_eval", ["classifierLast",
                                                                               "classifierLast2",
                                                                               "classifierAll",
                                                                               "confidence"]),
                                  Param("algo.params.evaluator.variance_explained", 0.9),
                                  Param("algo", "fedseq"),
                                  Param("do_train", False),
                                  Param("algo.params.clustering.classnames_eval",
                                        ["RandomClusterMaker", "GreedyClusterMaker", "KMeansClusterMaker"]),
                                  Param("algo.params.clustering.measures_eval",
                                        ["gini", "kullback", "cosine", "wasserstein"]),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [400, 800, 1000],
                                                   "max_clients": [7, 11, 13]}),
                                  MultiParam.key("algo.params.evaluator.epochs", [10, 20]),
                                  MultiParam.dict("algo.params.evaluator.model_info",
                                                  {"type": ["lenet"], "classname": ["LeNet"],
                                                   "pretrained": [False], "feature_extract": [False]}),
                                  runner_options={"--time": "01:00:00"}
                                  ),
        FedExperiment.from_params("FedSeq - clustering comparison mobilenetv2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("do_train", False),
                                  Param("algo.params.clustering.classnames_eval",
                                        ["RandomClusterMaker", "GreedyClusterMaker", "KMeansClusterMaker"]),
                                  Param("algo.params.clustering.measures_eval",
                                        ["gini", "kullback", "cosine", "wasserstein"]),
                                  Param("algo.params.evaluator.extract_eval", ["classifierLast", "confidence"]),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [400, 800, 1000],
                                                   "max_clients": [7, 11, 13]}),
                                  MultiParam.key("algo.params.evaluator.epochs", [10, 20]),
                                  MultiParam.dict("algo.params.evaluator.model_info",
                                                  {"type": ["mobilenetv2"], "classname": ["MobileNetV2"],
                                                   "pretrained": [True]}),
                                  MultiParam.key("algo.params.evaluator.model_info.feature_extract",
                                                 [True, False]),
                                  runner_options={"--time": "01:00:00"}
                                  ),
        FedExperiment.from_param_groups("FedSeq - best greedy-classifiers forgetting",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            Param("algo.params.training.check_forgetting", True),
                                            Param("algo.params.evaluator.extract", "classifierAll"),
                                            MultiParam.dict("algo.params.clustering",
                                                            {"min_examples": [800], "max_clients": [11],
                                                             "classname": ["GreedyClusterMaker"],
                                                             "measure": ["cosine"]
                                                             })
                                        ],
                                        runner_options={"--time": "4-00:00:00"}
                                        ),
        FedExperiment.from_param_groups("FedSeq - best greedy-confidence forgetting",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            Param("algo.params.training.check_forgetting", True),
                                            Param("algo.params.evaluator.extract", "confidence"),
                                            MultiParam.dict("algo.params.clustering",
                                                            {"min_examples": [800], "max_clients": [11],
                                                             "classname": ["GreedyClusterMaker"],
                                                             "measure": ["kullback"]
                                                             }),
                                        ],
                                        runner_options={"--time": "4-00:00:00"}
                                        ),
        FedExperiment.from_param_groups("FedSeq - best greedy-classifiers",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            Param("algo.params.evaluator.extract", "classifierAll"),
                                            MultiParam.dict("algo.params.clustering",
                                                            {"min_examples": [800], "max_clients": [11],
                                                             "classname": ["GreedyClusterMaker"],
                                                             "measure": ["cosine"]
                                                             }),
                                            MultiParam.key("algo.params.training.shuffle_sp_clients", [False, True])
                                        ],
                                        runner_options={"--time": "3-00:00:00"}
                                        ),
        FedExperiment.from_param_groups("FedSeq - best greedy-confidence",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            Param("algo.params.evaluator.extract", "confidence"),
                                            MultiParam.dict("algo.params.clustering",
                                                            {"min_examples": [800], "max_clients": [11],
                                                             "classname": ["GreedyClusterMaker"],
                                                             "measure": ["kullback"]
                                                             }),
                                            MultiParam.key("algo.params.training.shuffle_sp_clients", [False, True])
                                        ],
                                        runner_options={"--time": "3-00:00:00"}
                                        ),
        FedExperiment.from_param_groups("FedSeq - distillation best greedy-confidence",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            Param("loss", "kldistillation"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            MultiParam.key("loss.params.alpha", [0.25, 0.5, 0.75]),
                                            Param("algo.params.training.check_forgetting", True),
                                            Param("algo.params.training.shuffle_sp_clients", False),
                                            Param("algo.params.evaluator.extract", "confidence"),
                                            MultiParam.dict("algo.params.clustering",
                                                            {"min_examples": [800], "max_clients": [11],
                                                             "classname": ["GreedyClusterMaker"],
                                                             "measure": ["kullback"]
                                                             })
                                        ],
                                        runner_options={"--time": "5-00:00:00"}
                                        ),
        FedExperiment.from_param_groups("FedSeq - distillation best greedy-classifiers",
                                        "",
                                        [
                                            Param("dataset", "cifar10"),
                                            Param("n_round", 10000),

                                        ],
                                        [
                                            Param("dataset", "cifar100"),
                                            Param("n_round", 20000),
                                        ],
                                        shared_param_group=[
                                            Param("algo", "fedseq"),
                                            Param("loss", "kldistillation"),
                                            MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                            MultiParam.key("loss.params.alpha", [0.25, 0.5, 0.75]),
                                            Param("algo.params.training.check_forgetting", True),
                                            Param("algo.params.training.shuffle_sp_clients", False),
                                            Param("algo.params.evaluator.extract", "classifierAll"),
                                            MultiParam.dict("algo.params.clustering",
                                                            {"min_examples": [800], "max_clients": [11],
                                                             "classname": ["GreedyClusterMaker"],
                                                             "measure": ["cosine"]
                                                             })
                                        ],
                                        runner_options={"--time": "5-00:00:00"}
                                        ),
        FedExperiment.from_params("FedSeq - forgetting per class best greedy-classifier",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 10),
                                  Param("algo.params.training.shuffle_sp_clients", False),
                                  MultiParam.key("algo.params.training.average_model", [False, True]),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                  Param("algo.params.training.check_forgetting", True),
                                  Param("algo.params.evaluator.extract", "classifierAll"),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [800], "max_clients": [11],
                                                   "classname": ["GreedyClusterMaker"],
                                                   "measure": ["cosine"]
                                                   }),
                                  runner_options={"--time": "00:30:00"}
                                  ),
        FedExperiment.from_params("FedSeq - forgetting per class best greedy-confidence",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 10),
                                  Param("algo.params.training.shuffle_sp_clients", False),
                                  MultiParam.key("algo.params.training.average_model", [False, True]),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                  Param("algo.params.training.check_forgetting", True),
                                  Param("algo.params.evaluator.extract", "confidence"),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [800], "max_clients": [11],
                                                   "classname": ["GreedyClusterMaker"],
                                                   "measure": ["kullback"]
                                                   }),
                                  runner_options={"--time": "00:30:00"}
                                  ),
        FedExperiment.from_params("FedAvg - forgetting per class",
                                  "",
                                  Param("algo", "fedavg"),
                                  Param("n_round", 10),
                                  MultiParam.key("dataset", ["cifar10", "cifar100"]),
                                  MultiParam.key("common.alpha", [0, 0.2, 0.5]),
                                  runner_options={"--time": "00:30:00"}
                                  ),
    ]
    
    SHAKESPEARE EXPERIMENTS

    experiments = [
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy classifierLast C:0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "classifierLast"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["cosine", "wasserstein"]),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),

        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy classifierLast C:0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "classifierLast"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["cosine", "wasserstein"]),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),

        FedExperiment.from_params("FedSeq - runs comparison shakespeare - KMeans C:0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                  Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),
        
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - KMeans C:0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                  Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "01:20:00"}
                                  ),

        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy confidence C:0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "confidence"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),
        
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy confidence C:0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "confidence"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "01:20:00"}
                                  ),
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Random C 0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("algo.params.clustering.classname","RandomClusterMaker"),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:35:00"}
                                  ),
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Random C 0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("algo.params.clustering.classname","RandomClusterMaker"),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:50:00"}
                                  ),
        FedExperiment.from_params("FedAvg - runs shakespeare - C 0.1",
                                  "",
                                  Param("algo", "fedavg"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 1500),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.1),
                                  runner_options={"--time": "00:35:00"}
                                  ),
        FedExperiment.from_params("FedAvg - runs shakespeare - C 0.2",
                                  "",
                                  Param("algo", "fedavg"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 250),
                                  Param("dataset", "shakespeare_iid"),
                                  Param("algo.params.common.C", 0.2),
                                  runner_options={"--time": "00:50:00"}
                                  )
    ]

    EMNIST experiments 

    experiments = [
        FedExperiment.from_params("Centralized - runs comparison femnist - different lrs, weight decays",
                                  "",
                                  Param("algo", "centralized"),
                                  Param("device", "cuda:0"),
                                  Param("n_round", 100),
                                  MultiParam.key("algo.params.optim.args.lr", [5e-2, 1e-2, 5e-3, 1e-3]),
                                  MultiParam.key("algo.params.optim.args.weight_decay", [0, 1e-4]),
                                  Param("dataset", "emnist_niid"),
                                  Param("model", "emnist"),
                                  runner_options={"--time": "05:30:00"}
                                  )

    ]
    
    experiments = [
        FedExperiment.from_params("FedAvg - runs EMNIST - C 0.1",
                            "",
                            Param("algo", "fedavg"),
                            Param("device", "cuda:0"),
                            Param("model","emnist"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",0),
                            Param("n_round", 1500),
                            Param("dataset", "emnist_niid"),
                            Param("algo.params.common.C", 0.1),
                            runner_options={"--time": "9:00:00"}
                            ),
        FedExperiment.from_params("FedAvg - runs EMNIST - C 0.2",
                            "",
                            Param("algo", "fedavg"),
                            Param("device", "cuda:0"),
                            Param("model","emnist"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",0),
                            Param("n_round", 1500),
                            Param("dataset", "emnist_niid"),
                            Param("algo.params.common.C", 0.2),
                            runner_options={"--time": "11:00:00"}
                            )
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy classifierLast C:0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                        ["cosine", "wasserstein"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [1030, 2060, 4120],
                                                   "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "09:00:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy classifierLast C:0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                    ["cosine", "wasserstein"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "11:00:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - KMeans C:0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "09:00:00"}
                                ),
        
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - KMeans C:0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "11:00:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy confidence C:0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "09:00:00"}
                                ),
        
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy confidence C:0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "11:00:00"}
                                ),
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Random C 0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.clustering.classname","RandomClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "09:00:00"}
                                ),
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Random C 0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.clustering.classname","RandomClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "11:00:00"}
                                ),
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy classifierLast C:0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                        ["cosine", "wasserstein"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [1030, 2060, 4120],
                                                   "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "10:30:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy classifierLast C:0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                    ["cosine", "wasserstein"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "13:00:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - KMeans C:0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "10:30:00"}
                                ),
        
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - KMeans C:0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "13:00:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy confidence C:0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "10:30:00"}
                                ),
        
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Greedy confidence C:0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "13:00:00"}
                                ),
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Random C 0.1",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.clustering.classname","RandomClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.1),
                                Param("algo.params.common.K", 3500),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "08:30:00"}
                                ),
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - Random C 0.2",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.clustering.classname","RandomClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                MultiParam.dict("algo.params.clustering",
                                                {"min_examples": [1030, 2060, 4120],
                                                "max_clients": [6, 11, 21]}),
                                runner_options={"--time": "11:00:00"}
                                ),
        FedExperiment.from_params("FedAvg - runs EMNIST - C 0.2",
                            "",
                            Param("algo", "fedavg"),
                            Param("device", "cuda:0"),
                            Param("model","emnist"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",0),
                            Param("n_round", 1500),
                            Param("dataset", "emnist_niid"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.common.K", 3500),
                            runner_options={"--time": "12:00:00"}
                            )
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - runs comparison EMNIST - greedy C:02 wassertein 6 rerun",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                Param("algo.params.clustering.measure","wasserstein"),    
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.clustering.min_examples", 1030),
                                Param("algo.params.clustering.max_clients", 6),
                                runner_options={"--time": "13:00:00"}
                                ),

        FedExperiment.from_params("FedSeq - runs comparison EMNIST - greedy C:02 cosine 21 rerun",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                Param("algo.params.clustering.measure","cosine"),    
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.clustering.min_examples", 4120),
                                Param("algo.params.clustering.max_clients", 21),
                                runner_options={"--time": "13:00:00"}
                                ),     
        
    ]
    
    experiments = [
        
        FedExperiment.from_params("FedSeq Inter Shakespeare C:0.2 niid",
                                  "",
                                Param("algo", "fedseq_inter"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("algo.params.evaluator.optim.args.lr",1),
                                Param("algo.params.evaluator.optim.args.weight_decay", 1e-4),
                                Param("algo.params.evaluator.extract", "classifierLast"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                Param("algo.params.clustering.measure","wasserstein"),    
                                Param("dataset", "shakespeare_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                Param("algo.params.clustering.min_examples", 8000),
                                Param("algo.params.clustering.max_clients", 5),
                                runner_options={"--time": "00:42:00", "--mem":"4GB"},
                                ),
        
        FedExperiment.from_params("FedSeq Inter Shakespeare C:0.2 iid",
                                  "",
                                Param("algo", "fedseq_inter"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("algo.params.evaluator.optim.args.lr",1),
                                Param("algo.params.evaluator.optim.args.weight_decay", 1e-4),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                Param("algo.params.clustering.measure","gini"),    
                                Param("dataset", "shakespeare_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                Param("algo.params.clustering.min_examples", 8000),
                                Param("algo.params.clustering.max_clients", 5),
                                runner_options={"--time": "00:42:00", "--mem":"4GB"},
                                ),
        
        FedExperiment.from_params("FedSeq Inter EMNIST C:0.2 niid",
                                  "",
                                Param("algo", "fedseq_inter"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay", 0),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                Param("algo.params.clustering.min_examples", 4120),
                                Param("algo.params.clustering.max_clients", 21),
                                runner_options={"--time": "11:00:00", "--mem":"34GB"},
                                ),
        
        FedExperiment.from_params("FedSeq Inter EMNIST C:0.2 iid",
                                  "",
                                Param("algo", "fedseq_inter"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("algo.params.evaluator.optim.args.lr",0.01),
                                Param("algo.params.evaluator.optim.args.weight_decay", 0),
                                Param("algo.params.evaluator.extract", "confidence"),
                                Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                Param("dataset", "emnist_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                Param("algo.params.clustering.min_examples", 4120),
                                Param("algo.params.clustering.max_clients", 21),
                                runner_options={"--time": "11:00:00", "--mem":"34GB"},
                                ),

        
        FedExperiment.from_params("SCAFFOLD Shakespeare C:0.2 niid",
                                  "",
                                Param("algo", "scaffold"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("dataset", "shakespeare_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                runner_options={"--time": "00:42:00", "--mem":"4GB"},
                                ),
        
        FedExperiment.from_params("SCAFFOLD Shakespeare C:0.2 iid",
                                  "",
                                Param("algo", "scaffold"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("dataset", "shakespeare_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                runner_options={"--time": "00:42:00", "--mem":"4GB"},
                                ),
        
        FedExperiment.from_params("SCAFFOLD EMNIST C:0.2 niid",
                                  "",
                                Param("algo", "scaffold"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                runner_options={"--time": "11:00:00", "--mem":"34GB"},
                                ),
        
        FedExperiment.from_params("SCAFFOLD EMNIST C:0.2 iid",
                                  "",
                                Param("algo", "scaffold"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                runner_options={"--time": "11:00:00", "--mem":"34GB"},
                                ),

        
        FedExperiment.from_params("FedDyn Shakespeare C:0.2 niid",
                                  "",
                                Param("algo", "feddyn"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("dataset", "shakespeare_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                                runner_options={"--time": "00:42:00", "--mem":"6GB"},
                                ),
        
        FedExperiment.from_params("FedDyn Shakespeare C:0.2 iid",
                                  "",
                                Param("algo", "feddyn"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("dataset", "shakespeare_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                                runner_options={"--time": "00:42:00", "--mem":"6GB"},
                                ),
        
        FedExperiment.from_params("FedDyn EMNIST C:0.2 niid",
                                  "",
                                Param("algo", "feddyn"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                                runner_options={"--time": "11:00:00", "--mem":"40GB"},
                                ),
        
        FedExperiment.from_params("FedDyn EMNIST C:0.2 iid",
                                  "",
                                Param("algo", "feddyn"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                                runner_options={"--time": "11:00:00", "--mem":"40GB"},
                                ),

       
        FedExperiment.from_params("FedProx Shakespeare C:0.2 niid",
                                  "",
                                Param("algo", "fedprox"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("dataset", "shakespeare_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                MultiParam.key("algo.params.optim.args.mu",
                                        [0.01, 0.001, 0.0001]),
                                runner_options={"--time": "00:42:00", "--mem":"4GB"},
                                ),
        
        FedExperiment.from_params("FedProx Shakespeare C:0.2 iid",
                                  "",
                                Param("algo", "fedprox"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("model","shakespeare"),
                                Param("algo.params.optim.args.lr",1),
                                Param("algo.params.optim.args.weight_decay",1e-4),
                                Param("dataset", "shakespeare_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                MultiParam.key("algo.params.optim.args.mu",
                                        [0.01, 0.001, 0.0001]),
                                runner_options={"--time": "00:42:00", "--mem":"4GB"},
                                ),
        
        FedExperiment.from_params("FedProx EMNIST C:0.2 niid",
                                  "",
                                Param("algo", "fedprox"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                MultiParam.key("algo.params.optim.args.mu",
                                        [0.01, 0.001, 0.0001]),
                                runner_options={"--time": "11:00:00", "--mem":"34GB"},
                                ),
        
        FedExperiment.from_params("FedProx EMNIST C:0.2 iid",
                                  "",
                                Param("algo", "fedprox"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_iid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                MultiParam.key("algo.params.optim.args.mu",
                                        [0.01, 0.001, 0.0001]),
                                runner_options={"--time": "11:00:00", "--mem":"34GB"},
                                ),
        
    ]
    
    experiments = [
        
        FedExperiment.from_params("SCAFFOLD EMNIST C:0.2 niid",
                                  "",
                                Param("algo", "scaffold"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("model","emnist"),
                                Param("algo.params.optim.args.lr",0.01),
                                Param("algo.params.optim.args.weight_decay",0),
                                Param("dataset", "emnist_niid"),
                                Param("algo.params.common.C", 0.2),
                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                runner_options={"--time": "38:00:00", "--mem":"34GB"},
                                ),
        


       
        
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - Cifar10 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.2),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "20:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar10 - ClassifierLast Greedy Cosine",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.2),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "20:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.2),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - ClassifierLast Greedy Cosine",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.2),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),

                             
    ]

    experiments = [ 
        
        FedExperiment.from_params("Stackoverflow - centralized 1",
                                            "",
                                          Param("algo", "centralized"),
                                          Param("algo.params.batch_size", "16"),
                                          Param("device", "cuda:0"),
                                          Param("n_round", 50),
                                          Param("do_train", True),
                                          Param("model","soverflow"),
                                          Param("algo.params.optim.args.lr",0.316227766),
                                          Param("algo.params.optim.args.weight_decay",0),
                                          Param("dataset", "soverflow_niid"),
                                          runner_options={"--time": "56:00:00","--mem":"20GB"}
                                          )
    ]
    
    
    experiments = [ 
        FedExperiment.from_params("fedavg - shakespeare niid - whole dataset - C=0.1, 0.2",
                            "",
                            Param("algo", "fedavg"),
                            Param("device", "cuda:0"),
                            Param("n_round", 250),
                            Param("algo.params.optim.args.lr",1),
                            Param("algo.params.optim.args.weight_decay",1e-4),
                            Param("dataset", "shakespeare_niid"),
                            Param("algo.params.common.B", 100),
                            Param("algo.params.common.K", 1102),
                            Param("model","shakespeare"),
                            Param("algo.params.common.C",0.2),
                            runner_options={"--time": "15:00:00"}
                            ),
        FedExperiment.from_params("fedseq - shakespeare niid - whole dataset - C=0.1, 0.2",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("n_round", 250),
                            Param("algo.params.optim.args.lr",1),
                            Param("algo.params.optim.args.weight_decay",1e-4),
                            Param("algo.params.evaluator.optim.args.lr",1),
                            Param("algo.params.evaluator.optim.args.weight_decay",1e-4),
                            Param("dataset", "shakespeare_niid"),
                            Param("algo.params.common.B", 100),
                            Param("algo.params.common.K", 1102),
                            Param("model","shakespeare"),
                            MultiParam.key("algo.params.common.C",
                                        [0.1, 0.2]),
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 5000),#
                            Param("algo.params.clustering.max_clients", 13),#
                            runner_options={"--time": "20:00:00"}
                            ),
    ]
    
    experiments = [ 
        FedExperiment.from_params("fedseq - cifar 500 0.5 - C=0.1 dynamic kmeans",
                            "",
                            Param("algo", "fedseq_dynamic"),
                            Param("device", "cuda:0"),
                            Param("n_round", 2500),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("algo.params.evaluator.optim.args.lr",0.01),
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.K", 500),
                            Param("model","lenet"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.measure","cosine"),
                            MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [400, 800],
                                                   "max_clients": [7, 11]}),
                            runner_options={"--time": "10:00:00"}
                            ),
          FedExperiment.from_params("fedseq - cifar 500 0.5 - C=0.1 static kmeans",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("n_round", 2500),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("algo.params.evaluator.optim.args.lr",0.01),
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.K", 500),
                            Param("model","lenet"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.classname","KMeansClusterMaker"),
                            MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [400, 800],
                                                   "max_clients": [7, 11]}),
                            runner_options={"--time": "10:00:00"}
                            ),
    ]
    
    experiments = [ 
        FedExperiment.from_params("fedseq - cifar 500 0.5 - C=0.1 dynamic kmeans - way higher prob of swapping",
                            "",
                            Param("algo", "fedseq_dynamic"),
                            Param("device", "cuda:0"),
                            Param("n_round", 2500),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("algo.params.evaluator.optim.args.lr",0.01),
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.K", 500),
                            Param("model","lenet"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.measure","cosine"),
                            MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [400, 800],
                                                   "max_clients": [7, 11]}),
                            runner_options={"--time": "10:00:00"}
                            ),
    ]
    
    experiments = [ 
        
        FedExperiment.from_params("Stackoverflow - fedavg ",
                                            "",
                                        Param("algo", "fedavg"),
                                        Param("device", "cuda:0"),
                                        Param("n_round", 1500),
                                        Param("do_train", True),
                                        Param("model","soverflow"),
                                        Param("algo.params.optim.args.lr",0.316227766),
                                        Param("algo.params.optim.args.weight_decay",0),
                                        Param("dataset", "soverflow_niid"),
                                        Param("algo.params.common.B", 16),
                                        Param("algo.params.common.K", 40000),
                                        Param("algo.params.common.C", 0.01),
                                        runner_options={"--time": "24:00:00","--mem":"16GB"}
                                        ),
        
    ]
    
    experiments = [ 
        FedExperiment.from_params("Stackoverflow - fedseq random ",
                                            "",
                                        Param("algo", "fedseq"),
                                        Param("device", "cuda:0"),
                                        Param("n_round", 1500),
                                        Param("do_train", True),
                                        Param("model","soverflow"),
                                        Param("algo.params.optim.args.lr",0.316227766),
                                        Param("algo.params.optim.args.weight_decay",0),
                                        Param("dataset", "soverflow_niid"),
                                        Param("algo.params.common.B", 16),
                                        Param("algo.params.common.K", 40000),
                                        Param("algo.params.common.C", 0.01),
                                        Param("algo.params.evaluator.optim.args.lr",0.316227766),
                                        Param("algo.params.evaluator.optim.args.weight_decay",0),
                                        Param("algo.params.clustering.classname","RandomClusterMaker"),
                                        Param("algo.params.clustering.min_examples",8500),
                                        Param("algo.params.clustering.max_clients",25),
                                        runner_options={"--time": "28:00:00","--mem":"18GB"}
                                        ),
        FedExperiment.from_params("Stackoverflow - fedseq save cluster from (confidence) Greedy kullback",
                                      "",
                                    Param("algo", "fedseq"),
                                    Param("algo.params.evaluator.extract", "confidence"),
                                    Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                    Param("algo.params.clustering.measure","kullback"),
                                    Param("algo.params.evaluator.optim.args.lr",0.316227766),
                                    Param("algo.params.evaluator.optim.args.weight_decay",0),
                                    Param("algo.params.evaluator.epochs",5),
                                    Param("algo.params.center_server.classname", "FedAvgCenterServer"),
                                    Param("algo.params.save_representers",False),
                                    Param("algo.params.save_superclients",True),
                                    Param("algo.params.evaluator.precomputed","./datasets/stackoverflow/representers/confidence.pkl"),
                                    Param("algo.params.common.C", "0.01"),
                                    Param("algo.params.common.K", "40000"),
                                    MultiParam.dict("algo.params.clustering",
                                                    {"min_examples": [8500],
                                                    "max_clients": [25]}),
                                    Param("algo.params.common.B", "16"),
                                    Param("device", "cuda:0"),
                                    Param("wandb.client_datasets",False),
                                    Param("wandb.superclient_datasets",False),
                                    Param("n_round", 1500),
                                    Param("do_train", False),
                                    Param("model","soverflow"),
                                    Param("algo.params.optim.args.lr",0.316227766),
                                    Param("algo.params.optim.args.weight_decay",0),
                                    Param("dataset", "soverflow_niid"),
                                    runner_options={"--time": "80:00:00","--mem":"30GB"}
                                    ),
        
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - Cifar10 - task2vec",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            MultiParam.key("algo.params.common.C", [0.1, 0.2]),
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.evaluator.task2vec.method",
                                        ['montecarlo', 'variational']),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "24:00:00"}
                            ),
        
        FedExperiment.from_params("FedSeq - Cifar100 - task2vec",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            MultiParam.key("algo.params.common.C", [0.1, 0.2]),
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.evaluator.task2vec.method",
                                        ['montecarlo', 'variational']),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "48:00:00"}
                            ),
        

                             
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - Cifar10 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "18:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar10 - ClassifierLast Greedy Cosine",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "18:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - ClassifierLast Greedy Cosine",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),

                             
    ]
    
    experiments = [
        FedExperiment.from_params("FedSeq - Cifar10 - task2vec C=0.1 montecarlo resnet34 - alpha 0.5",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C",  0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.evaluator.task2vec.method",
                                        'montecarlo'),
                            Param("algo.params.evaluator.task2vec.probe_network",
                                        'resnet34'),
                            Param("algo.params.common.alpha",
                            0.5),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "24:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar10 - task2vec C=0.1 variational - alpha 0.2 and 0.5",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C",  0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.evaluator.task2vec.method",
                                        'variational'),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            MultiParam.key("algo.params.common.alpha",
                            [0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "24:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar10 - task2vec C=0.2 - alpha 0.2 and 0.5",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C",  0.2),
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.evaluator.task2vec.method",
                                        ['montecarlo', 'variational']),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            MultiParam.key("algo.params.common.alpha",
                            [0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "24:00:00"}
                            ),
        
        FedExperiment.from_params("FedSeq - Cifar100 - task2vec - alpha 0.2 and 0.5",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            MultiParam.key("algo.params.common.C", [0.1, 0.2]),
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.evaluator.task2vec.method",
                                        ['montecarlo', 'variational']),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha",
                                        [0.2, 0.5]),
                            runner_options={"--time": "48:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 C=0.1- task2vec - alpha 0 failed ones",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.evaluator.task2vec.method",
                                        'variational'),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.alpha",
                                        0),
                            runner_options={"--time": "48:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 C=0.2- task2vec MC - alpha 0 failed ones",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.evaluator.task2vec.method",
                                        'montecarlo'),
                            MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                        ['resnet18', 'resnet34']),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.alpha",
                                        0),
                            runner_options={"--time": "48:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 C=0.2- task2vec Var - alpha 0 failed ones",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "task2vec"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","normalized_cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.evaluator.task2vec.method",
                                        'variational'),
                            Param("algo.params.evaluator.task2vec.probe_network",
                                        'resnet34'),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.alpha",
                                        0),
                            runner_options={"--time": "48:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar10 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "18:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar10 - ClassifierLast Greedy Cosine",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "18:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - ClassifierLast Greedy Cosine",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),
        
        

                             
    ]
    
    
    experiments = [
        FedExperiment.from_params("FedSeqInter - Cifar10 - confidence kullback - c=0.1",
                            "",
                            Param("algo", "fedseq_inter"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C",  0.1),
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",  [0,0.2,0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "24:00:00"}
                            ),
        FedExperiment.from_params("FedSeqInter - Cifar10 - classifierlast cosine - c=0.1",
                            "",
                            Param("algo", "fedseq_inter"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C",  0.1),
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",  [0,0.2,0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "24:00:00"}
                            ),

        FedExperiment.from_params("FedSeqInter - Cifar100 - confidence kullback - c=0.1",
                            "",
                            Param("algo", "fedseq_inter"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha",
                                        [0,0.2, 0.5]),
                            runner_options={"--time": "48:00:00"}
                            ),
        FedExperiment.from_params("FedSeqInter - Cifar100 - classifierLast cosine - c=0.1",
                            "",
                            Param("algo", "fedseq_inter"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "classifierLast"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","cosine"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha",
                                        [0,0.2, 0.5]),
                            runner_options={"--time": "48:00:00"}
                            ),
        
        FedExperiment.from_params("FedAvg - cifar10 c=0.1",
                            "",
                            Param("algo", "fedavg"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            runner_options={"--time": "20:00:00"}
                            ),
        FedExperiment.from_params("FedAvg - cifar100 c=0.1",
                            "",
                            Param("algo", "fedavg"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            runner_options={"--time": "45:00:00"}
                            ),
        FedExperiment.from_params("scaffold - cifar10 c=0.1",
                            "",
                            Param("algo", "scaffold"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            runner_options={"--time": "26:00:00"}
                            ),
        FedExperiment.from_params("scaffold - cifar100 c=0.1",
                            "",
                            Param("algo", "scaffold"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            runner_options={"--time": "50:00:00"}
                            ),
        FedExperiment.from_params("fedprox - cifar10 c=0.1",
                            "",
                            Param("algo", "fedprox"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            MultiParam.key("algo.params.optim.args.mu",
                                        [0.01, 0.001, 0.0001]),
                            runner_options={"--time": "26:00:00"}
                            ),
        FedExperiment.from_params("fedprox - cifar100 c=0.1",
                            "",
                            Param("algo", "fedprox"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            MultiParam.key("algo.params.optim.args.mu",
                                        [0.01, 0.001, 0.0001]),
                            runner_options={"--time": "50:00:00"}
                            ),
        FedExperiment.from_params("feddyn - cifar10 c=0.1",
                            "",
                            Param("algo", "feddyn"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                            runner_options={"--time": "26:00:00"}
                            ),
        
        FedExperiment.from_params("feddyn - cifar100 c=0.2 alpha=0",
                            "",
                            Param("algo", "feddyn"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.2),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            Param("algo.params.common.alpha", 0),
                            MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                            runner_options={"--time": "50:00:00"}
                            ),


                            
        
        
        
          
    ]
    
    experiments = [ 
        FedExperiment.from_params("FedSeq - Cifar10 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","scipy_kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 10000),
                            Param("dataset", "cifar10"),
                            MultiParam.key("algo.params.common.C", [0.1, 0.2]),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "18:00:00"}
                            ),
        FedExperiment.from_params("FedSeq - Cifar100 - Confidence Greedy Kullback",
                            "",
                            Param("algo", "fedseq"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr", 0.01),#
                            Param("algo.params.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.optim.args.lr",0.01),#
                            Param("algo.params.evaluator.optim.args.weight_decay",4e-4),#
                            Param("algo.params.evaluator.extract", "confidence"),
                            Param("algo.params.clustering.classname","GreedyClusterMaker"),
                            Param("algo.params.clustering.measure","scipy_kullback"),
                            Param("algo.params.clustering.min_examples", 800),#
                            Param("algo.params.clustering.max_clients", 11),#
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            MultiParam.key("algo.params.common.C", [0.1, 0.2]),#
                            Param("algo.params.common.K", 500),
                            MultiParam.key("algo.params.common.alpha",
                                        [0, 0.2, 0.5]),
                            Param("algo.params.common.B", 64),
                            runner_options={"--time": "40:00:00"}
                            ),
    ]
    
    experiments = [ 
        FedExperiment.from_params("FedSeq - shakespeare - task2vec",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                Param("algo.params.clustering.measure",
                                    "normalized_cosine"),
                                MultiParam.key("dataset", ["shakespeare_niid","shakespeare_iid"]),
                                MultiParam.key("algo.params.common.C", [0.1, 0.2]),
                                Param("algo.params.clustering.min_examples", "8000"),
                                Param("algo.params.clustering.max_clients", "5"),
                                Param("algo.params.common.K", 100),
                                MultiParam.key("algo.params.evaluator.task2vec.method",
                                            ['montecarlo', 'variational']),
                                Param("algo.params.evaluator.task2vec.probe_network",
                                            'rnn'),
                                Param("algo.params.common.B", 100),
                                Param("algo.params.optim.args.lr", 1),#
                                Param("algo.params.optim.args.weight_decay",1e-4),#
                                Param("algo.params.evaluator.optim.args.lr",1),#
                                Param("algo.params.evaluator.optim.args.weight_decay",1e-4),#
                                runner_options={"--time": "02:30:00"}
                                ),
        
        
    ]
    
    experiments = [ 
        FedExperiment.from_params("feddyn - cifar10 c=0.1",
                            "",
                            Param("algo", "feddyn"),
                            Param("device", "cuda:0"),
                            Param("model","lenet"),
                            Param("algo.params.optim.args.lr",0.01),
                            Param("algo.params.optim.args.weight_decay",4e-4),
                            Param("n_round", 20000),
                            Param("dataset", "cifar100"),
                            Param("algo.params.common.C", 0.1),
                            Param("algo.params.common.K", 500),
                            Param("algo.params.common.B", 64),
                            MultiParam.key("algo.params.common.alpha", [0, 0.2, 0.5]),
                            MultiParam.key("algo.params.alpha",
                                        [0.001, 0.01, 0.015]),
                            runner_options={"--time": "50:00:00"}
                            ),
        
        
    ]
    """
    """
    experiments = [ 
        FedExperiment.from_params("FedSeq - femnist - task2vec",
                                  "",
                                Param("algo", "fedseq"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                Param("algo.params.clustering.measure",
                                    "normalized_cosine"),
                                MultiParam.key("dataset", ["emnist_niid","emnist_iid"]),
                                Param("model", "emnist"),
                                MultiParam.key("algo.params.common.C", [0.1, 0.2, 0.05, 0.02]),
                                Param("algo.params.clustering.min_examples", "4120"),
                                Param("algo.params.clustering.max_clients", "21"),
                                Param("algo.params.common.K", 3500),
                                MultiParam.key("algo.params.evaluator.task2vec.method",
                                            ['montecarlo', 'variational']),
                                MultiParam.key("algo.params.evaluator.task2vec.probe_network",
                                            ['resnet18', 'resnet34']),
                                Param("algo.params.common.B", 20),
                                Param("algo.params.optim.args.lr", 0.01),#
                                Param("algo.params.optim.args.weight_decay",0),#
                                Param("algo.params.evaluator.optim.args.lr",0.01),#
                                Param("algo.params.evaluator.optim.args.weight_decay",0),#
                                runner_options={"--time": "20:00:00"}
                                ),
        
        
    ]
    experiments = [ 
        FedExperiment.from_params("logFSParallel - femnist subset",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("device", "cuda:0"),
                                Param("n_round", 500),
                                Param("do_train", True),
                                Param("algo.params.evaluator.extract", "classDistribution"),
                                Param("algo.params.clustering.classname", "ICGClusterMaker"),
                                Param("algo.params.clustering.measure", "euclidean"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                Param("dataset", "emnist_niid_subset"),
                                Param("model", "emnist"),
                                Param("algo.params.common.K", 368),
                                Param("algo.params.common.B", 5),
                                Param("algo.params.common.C", 0.3),
                                Param("algo.params.optim.args.lr", 0.01),
                                Param("algo.params.optim.args.weight_decay", 0),
                                Param("algo.params.growth_func", "log"),
                                MultiParam.key("algo.params.alpha_growth", [1, 2, 4, 6, 8, 10]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 30, 40, 50]),
                                runner_options={"--time": "05:00:00"}
                                ),
        FedExperiment.from_params("linFSParallel - femnist subset",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("device", "cuda:0"),
                                Param("n_round", 500),
                                Param("do_train", True),
                                Param("algo.params.evaluator.extract", "classDistribution"),
                                Param("algo.params.clustering.classname", "ICGClusterMaker"),
                                Param("algo.params.clustering.measure", "euclidean"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                Param("dataset", "emnist_niid_subset"),
                                Param("model", "emnist"),
                                Param("algo.params.common.K", 368),
                                Param("algo.params.common.B", 5),
                                Param("algo.params.common.C", 0.3),
                                Param("algo.params.optim.args.lr", 0.01),
                                Param("algo.params.optim.args.weight_decay", 0),
                                Param("algo.params.growth_func", "linear"),
                                MultiParam.key("algo.params.alpha_growth", [1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 10e-2]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 30, 40, 50]),
                                runner_options={"--time": "05:00:00"}
                                ),
        FedExperiment.from_params("expFSParallel - femnist subset",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("device", "cuda:0"),
                                Param("n_round", 500),
                                Param("do_train", True),
                                Param("algo.params.evaluator.extract", "classDistribution"),
                                Param("algo.params.clustering.classname", "ICGClusterMaker"),
                                Param("algo.params.clustering.measure", "euclidean"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                Param("dataset", "emnist_niid_subset"),
                                Param("model", "emnist"),
                                Param("algo.params.common.K", 368),
                                Param("algo.params.common.B", 5),
                                Param("algo.params.common.C", 0.3),
                                Param("algo.params.optim.args.lr", 0.01),
                                Param("algo.params.optim.args.weight_decay", 0),
                                Param("algo.params.growth_func", "exp"),
                                MultiParam.key("algo.params.alpha_growth", [0.6e-2, 1.2e-2, 1.8e-2, 2.4e-2, 3e-2, 3.6e-2]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 30, 40, 50]),
                                runner_options={"--time": "05:00:00"}
                                ),
        
        
    ]

    
    
    experiments = [ 
        FedExperiment.from_params("Shakespeare - log",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("dataset", "shakespeare_niid"),
                                Param("model", "shakespeare"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("do_train", True),
                                
                                Param("algo.params.optim.args.lr", 1),
                                Param("algo.params.optim.args.weight_decay", 0.0001),

                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                Param("algo.params.common.C", 0.2),

                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.evaluator.task2vec.method", "montecarlo"),
                                Param("algo.params.evaluator.task2vec.probe_network", "charGPT"),
                                MultiParam.key("algo.params.clustering.classname", ["ICGClusterMaker", "RandomClusterMaker"]),
                                Param("algo.params.clustering.measure", "normalized_cosine"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                    
                                Param("algo.params.growth_func", "log"),
                                MultiParam.key("algo.params.alpha_growth", [0.5, 1, 2]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 25]),
                                runner_options={"--time": "12:00:00"}
                                ),
         FedExperiment.from_params("Shakespeare - linear",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("dataset", "shakespeare_niid"),
                                Param("model", "shakespeare"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("do_train", True),
                                
                                Param("algo.params.optim.args.lr", 1),
                                Param("algo.params.optim.args.weight_decay", 0.0001),

                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                Param("algo.params.common.C", 0.2),

                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.evaluator.task2vec.method", "montecarlo"),
                                Param("algo.params.evaluator.task2vec.probe_network", "charGPT"),
                                MultiParam.key("algo.params.clustering.classname", ["ICGClusterMaker", "RandomClusterMaker"]),
                                Param("algo.params.clustering.measure", "normalized_cosine"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                    
                                Param("algo.params.growth_func", "linear"),
                                MultiParam.key("algo.params.alpha_growth", [0.005, 0.01, 0.02]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 25]),
                                runner_options={"--time": "12:00:00"}
                                ),
         FedExperiment.from_params("Shakespeare - exp",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("dataset", "shakespeare_niid"),
                                Param("model", "shakespeare"),
                                Param("device", "cuda:0"),
                                Param("n_round", 250),
                                Param("do_train", True),
                                
                                Param("algo.params.optim.args.lr", 1),
                                Param("algo.params.optim.args.weight_decay", 0.0001),

                                Param("algo.params.common.K", 100),
                                Param("algo.params.common.B", 100),
                                Param("algo.params.common.C", 0.2),

                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.evaluator.task2vec.method", "montecarlo"),
                                Param("algo.params.evaluator.task2vec.probe_network", "charGPT"),
                                MultiParam.key("algo.params.clustering.classname", ["ICGClusterMaker", "RandomClusterMaker"]),
                                Param("algo.params.clustering.measure", "normalized_cosine"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                    
                                Param("algo.params.growth_func", "exp"),
                                MultiParam.key("algo.params.alpha_growth", [0.006, 0.012, 0.018]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 25]),
                                runner_options={"--time": "12:00:00"}
                                ),
        
        
    
    
    
    
    
    
    
    ]
    """
    experiments = [
        #THESE ARE THE ACTUAL EXPERIMENTS FOR FEMNIST FOR FEDSEQPARALLEL!!!
        FedExperiment.from_params("FEMNIST - log",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("dataset", "emnist_niid"),
                                Param("model", "emnist"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("do_train", True),
                                
                                Param("algo.params.optim.args.lr", 0.01),
                                Param("algo.params.optim.args.weight_decay", 0),

                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                Param("algo.params.common.C", 0.2),

                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.evaluator.task2vec.method", "montecarlo"),
                                Param("algo.params.evaluator.task2vec.probe_network", "resnet18"),
                                Param("algo.params.clustering.classname", "ICGClusterMaker"),
                                Param("algo.params.clustering.measure", "normalized_cosine"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                    
                                Param("algo.params.growth_func", "log"),
                                MultiParam.key("algo.params.alpha_growth", [10, 15, 20]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 25]),
                                runner_options={"--time": "30:00:00"}
                                ),
         FedExperiment.from_params("FEMNIST - linear",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("dataset", "emnist_niid"),
                                Param("model", "emnist"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("do_train", True),
                                
                                Param("algo.params.optim.args.lr", 0.01),
                                Param("algo.params.optim.args.weight_decay", 0),

                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                Param("algo.params.common.C", 0.2),

                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.evaluator.task2vec.method", "montecarlo"),
                                Param("algo.params.evaluator.task2vec.probe_network", "resnet18"),
                                Param("algo.params.clustering.classname", "ICGClusterMaker"),
                                Param("algo.params.clustering.measure", "normalized_cosine"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                    
                                Param("algo.params.growth_func", "linear"),
                                MultiParam.key("algo.params.alpha_growth", [0.1, 0.15, 0.2]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 25]),
                                runner_options={"--time": "30:00:00"}
                                ),
         FedExperiment.from_params("FEMNIST - exp",
                                  "",
                                Param("algo", "fedseq_parallel"),
                                Param("dataset", "emnist_niid"),
                                Param("model", "emnist"),
                                Param("device", "cuda:0"),
                                Param("n_round", 1500),
                                Param("do_train", True),
                                
                                Param("algo.params.optim.args.lr", 0.01),
                                Param("algo.params.optim.args.weight_decay", 0),

                                Param("algo.params.common.K", 3500),
                                Param("algo.params.common.B", 20),
                                Param("algo.params.common.C", 0.2),

                                Param("algo.params.evaluator.extract", "task2vec"),
                                Param("algo.params.evaluator.task2vec.method", "montecarlo"),
                                Param("algo.params.evaluator.task2vec.probe_network", "resnet18"),
                                Param("algo.params.clustering.classname", "ICGClusterMaker"),
                                Param("algo.params.clustering.measure", "normalized_cosine"),
                                Param("algo.params.clustering.collect_time_statistics", True),
                                    
                                Param("algo.params.growth_func", "exp"),
                                MultiParam.key("algo.params.alpha_growth", [0.002, 0.003, 0.004]),
                                MultiParam.key("algo.params.beta_growth", [5, 10, 20, 25]),
                                runner_options={"--time": "30:00:00"}
                                ),
    ]
    
    r = SlurmRunner(experiment_config.get("seed"), 0.11, train_time_overshoot=0.04,
                            default_params=train_defaults)
    for e in experiments:
        print(e)
        e.run('train.py', r)

    r.wait_all()


if __name__ == "__main__":
    main()
