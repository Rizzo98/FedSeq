from typing import Callable, Dict, List
from collections import OrderedDict
import hydra
import numpy as np
import os
from pathlib import Path
import pickle
from omegaconf import DictConfig
from scipy.stats import t
from src.utils import savepickle


def aggregate_superclient_statistics(cfg: DictConfig):
  #search for "examples_superclients*" files to aggregate
  path = Path(cfg.folder)
  num_statistics = sum([1 for f in path.glob('examples_superclients_*.pkl') if os.path.isfile(f)])
  if num_statistics < cfg.fit_iterations: 
    print(f"Less examples_superclient than fit_iterations: expected {cfg.fit_iterations}, found {num_statistics}")
    if num_statistics == 0: return

  results = []
  for path in path.glob('examples_superclients_*.pkl'):
    with open(str(path), "rb") as f:
      res = pickle.load(f)
      results.append(res)
  f.close()

  #reduce using a reduce function
  n_superclients = np.min([len(res) for res in results])
  averaged_results = []
  for i in range(n_superclients):
    vec_to_avg = []
    for j in range(num_statistics):
      vec_to_avg.append(results[j][i])
    mean_vec = np.zeros(len(vec_to_avg[0]), dtype=int)
    for v in vec_to_avg:
      mean_vec = mean_vec + v.astype(int)
    mean_vec = mean_vec // num_statistics
    averaged_results.append(mean_vec)
  savepickle(averaged_results, os.path.join(cfg.folder, "examples_superclients.pkl"))


#aggregates {accuracy, loss} for each fit
def aggregate_fits(cfg: DictConfig):
  #search for "result*" files to aggregate
  path = Path(cfg.folder)
  num_fit = sum([1 for f in path.glob('result_*.pkl') if os.path.isfile(f)])
  if num_fit < cfg.fit_iterations: 
    print(f"Less results than fit_iterations: expected {cfg.fit_iterations}, found {num_fit}")
    if num_fit == 0: return

  results = []
  for path in path.glob('result_*.pkl'):
    with open(str(path), "rb") as f:
      res = pickle.load(f)
      results.append(res)
  f.close()

  #reduce using a reduce function
  reduced = compute_final_result(results, cfg.result_aggregation_method)

  #calculate deviations
  factor = t.ppf(cfg.confidence + (1-cfg.confidence)/2, cfg.fit_iterations-1)/np.sqrt(cfg.fit_iterations)
  dev = reduce_results(results, lambda x: np.std(x, ddof=1), num_fit, factor)

  #save onto disk
  savepickle(reduced, os.path.join(cfg.folder, "result.pkl"))
  savepickle(dev, os.path.join(cfg.folder, "deviations.pkl"))

def models_distance(cfg: DictConfig):
  #search for models folder
  assert len(cfg.distance_metrics) > 0, "No metric given"
  path = Path(cfg.folder)
  for entry in path.glob('models*'):
    if entry.is_dir():
      models_distance_folder(entry, cfg)


#calculate pairwise distance of the clients' models
def models_distance_folder(folder, cfg: DictConfig):
  #search for models to aggregate
  num_models = sum([1 for f in folder.glob('*.pkl') if os.path.isfile(f)])
  if num_models == 0:
    print("No model found")
    return

  metrics_func = dict([(metric, eval(metric)) for metric in cfg.distance_metrics])
  models = []
  for path in folder.glob('*.pkl'):
    with open(str(path), "rb") as f:
      res = pickle.load(f)  
      models.append(res)
    f.close()

  distances = np.empty((num_models, num_models))
  np.fill_diagonal(distances, 0) #distance of x from itself is zero by def
  for metric, metric_func in metrics_func.items():
    for i in range(num_models):
      for j in range(i):
        distances[i][j] = distances[j][i] = metric_func(models[i].state_dict(), models[j].state_dict())
    savepickle(distances, os.path.join(str(folder), "distances", f"{metric}.pkl"))

def cosine_dist(model1_state_dict: OrderedDict, model2_state_dict: OrderedDict) -> float:
  v1 = np.concatenate([np.ravel(val.numpy()) for val in model1_state_dict.values()]).flatten()
  v2 = np.concatenate([np.ravel(val.numpy()) for val in model2_state_dict.values()]).flatten()
  prod = np.dot(v1, v2)
  normsProd = np.linalg.norm(v1)*np.linalg.norm(v2)
  return 1-prod/normsProd

def eucledian_dist(model1_state_dict: OrderedDict, model2_state_dict: OrderedDict) -> float:
  v1 = np.concatenate([np.ravel(val.numpy()) for val in model1_state_dict.values()]).flatten()
  v2 = np.concatenate([np.ravel(val.numpy()) for val in model2_state_dict.values()]).flatten()
  return np.linalg.norm(v1-v2)

def reduce_results(results, reducer: Callable[[list],list], res_len, factor=1) -> Dict[str, list]:
  dev={"loss": [], "accuracy": [], "time_elapsed": []}
  for r in range(res_len):
    accuracy_r = []
    loss_r = []
    time_elapsed_r = []
    for it in range(len(results)):
      loss_r.append(results[it]["loss"][r])
      accuracy_r.append(results[it]["accuracy"][r])
      time_elapsed_r.append(results[it]["time_elapsed"][r])
    dev["loss"].append(factor*reducer(loss_r))
    dev["accuracy"].append(factor*reducer(accuracy_r))
    dev["time_elapsed"].append(factor*reducer(time_elapsed_r))
  return dev

def compute_final_result(results: List[Dict[str, list]], method):
  #for safety: select min of lenght of vectors
  num_rounds = np.min([len(res["accuracy"]) for res in results])
  if method == "mean":
    return reduce_results(results, np.mean, num_rounds)
  else:
    raise NotImplementedError

#expected the folder containing the result to aggregate
@hydra.main(config_path="config/experiments", config_name="config")
def main(cfg: DictConfig):
    assert os.path.isdir(cfg.folder), f"{cfg.folder} is not a directory"
    assert cfg.fit_iterations > 1, "Less than two iterations, nothing to aggregate"

    tasks = [eval(f) for f in cfg.functions]
    for t in tasks:
      t(cfg)


if __name__ == "__main__":
    main()