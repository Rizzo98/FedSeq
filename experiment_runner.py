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
    """
    experiments = [
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy classifierLast C:0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "classifierLast"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["cosine", "wasserstein"]),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),

        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy classifierLast C:0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "classifierLast"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["cosine", "wasserstein"]),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),

        FedExperiment.from_params("FedSeq - runs comparison shakespeare - KMeans C:0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                  Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),
        
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - KMeans C:0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  MultiParam.key("algo.params.evaluator.extract", ["classifierLast","confidence"]),
                                  Param("algo.params.clustering.classname","KMeansClusterMaker"),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "01:20:00"}
                                  ),

        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy confidence C:0.1",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "confidence"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.1),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:45:00"}
                                  ),
        
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Greedy confidence C:0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  Param("algo.params.evaluator.extract", "confidence"),
                                  Param("algo.params.clustering.classname","GreedyClusterMaker"),
                                  MultiParam.key("algo.params.clustering.measure",
                                        ["gini", "kullback"]),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "01:20:00"}
                                  ),
        
        FedExperiment.from_params("FedSeq - runs comparison shakespeare - Random C 0.2",
                                  "",
                                  Param("algo", "fedseq"),
                                  Param("n_round", 250),
                                  Param("algo.params.clustering.classname","RandomClusterMaker"),
                                  Param("dataset", "shakespeare_niid"),
                                  Param("common.C", 0.2),
                                  MultiParam.dict("algo.params.clustering",
                                                  {"min_examples": [4000, 8000],
                                                   "max_clients": [3, 5]}),
                                  runner_options={"--time": "00:50:00"}
                                  )

    ]
    r: Runner = SlurmRunner(experiment_config.get("seed"), 0.11, train_time_overshoot=0.04,
                            default_params=train_defaults, defaults={"--mem": "4GB"})
    for e in experiments:
        print(e)
        e.run('train.py', r)

    r.wait_all()


if __name__ == "__main__":
    main()
