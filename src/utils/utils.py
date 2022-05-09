import copy
import math
import pickle
from functools import wraps
import logging
import os
import random
import json
from re import A
from runpy import run_module
import signal, sys
import time
from matplotlib.pyplot import step
import wandb
from contextlib import contextmanager
from typing import Union
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name: str, logger: Union[logging.Logger, None] = None):
    t0 = time.time()
    yield
    msg = f'[{name}] done in {time.time() - t0:.3f} s'
    if logger:
        logger.info(msg)
    else:
        print(msg)


def tail_recursive(func):
    self_func = [func]
    self_firstcall = [True]
    self_CONTINUE = [object()]
    self_argskwd = [None]

    @wraps(func)
    def _tail_recursive(*args, **kwd):
        if self_firstcall[0] == True:
            func = self_func[0]
            CONTINUE = self_CONTINUE
            self_firstcall[0] = False
            try:
                while True:
                    result = func(*args, **kwd)
                    if result is CONTINUE:  # update arguments
                        args, kwd = self_argskwd[0]
                    else:  # last call
                        return result
            finally:
                self_firstcall[0] = True
        else:  # return the arguments of the tail call
            self_argskwd[0] = args, kwd
            return self_CONTINUE

    return _tail_recursive


def savepickle(obj, path: str, open_options: str = "wb"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, open_options) as f:
        pickle.dump(obj, f)
    f.close()


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
                 tag_prefix=''):
        super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                         flush_secs=flush_secs, filename_suffix=filename_suffix)
        self.tag_prefix = tag_prefix

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        newTag = f"{self.tag_prefix}/{tag}"
        return super().add_scalar(newTag, scalar_value, global_step=global_step, walltime=walltime)


class WanDBSummaryWriter:
    def __init__(self, config) -> None:
        project_name, run_name = self._generate_project_run_name(config)
        if config.wandb.restart_from_run is None:
            run_id = wandb.util.generate_id()
            wandb.init(id=run_id, project=project_name, entity=config.wandb.entity, config ={}, tags=config.wandb.tags)
            wandb.run.name = run_name
            self.local_store = dict()
        else:
            print(f'Restarting from run {config.wandb.restart_from_run}!')
            wandb.init(project=project_name, entity=config.wandb.entity, id=config.wandb.restart_from_run,resume='must')
            run_path = config.wandb.entity+'/'+project_name+'/'+config.wandb.restart_from_run
            model = torch.load(wandb.restore('models/last_model.pt', run_path=run_path).name)
            self.restore_run = dict()
            self.restore_run['model_weight'] = model['weight']
            self.restore_run['resume_round'] = model['round']
            self.local_store = json.load(wandb.restore('objects/local_store.json', run_path=run_path))
        
    def _generate_project_run_name(self, config):
        project_name = ""
        iid_ness = ""
        if "_" in config.dataset.name:
            project_name = config.dataset.name.split('_')[0]
            if config.algo.type!='centralized':
                iid_ness = f'_split:{(config.dataset.name.split("_")[1]).upper()}'
        else:
            project_name = config.dataset.name
            if 'common' in config.algo.params:
                iid_ness = f'_alpha:{config.algo.params.common.alpha}'
        n_rounds = f'rounds:{config.n_round}'
        run_name = f'{config.algo.type}{iid_ness}_{n_rounds}'
        if config.algo.type!='centralized':
            participation = f'C:{config.algo.params.common.C}'
            run_name+=f'_{participation}'
            if 'clustering' in config.algo.params:
                cluster = f'cluster:{(config.algo.params.clustering.classname).replace("ClusterMaker", "")}'
                run_name+=f'_{cluster}'
                if config.algo.params.clustering.classname != 'RandomClusterMaker':
                    extract = f'extract:{config.algo.params.evaluator.extract}'
                    run_name+=f'_{extract}'
                    if config.algo.params.clustering.classname != 'KMeansClusterMaker':
                        measure = f'measure:{config.algo.params.clustering.measure}'
                        run_name+=f'_{measure}'
                max_clients = f'maxClients:{config.algo.params.clustering.max_clients}'
                run_name+=f'_{max_clients}'
        if config.device=='cpu':
            project_name = 'test-project'
        if config.wandb.run_suffix is not None:
            run_name+=f'_repetition:{config.wandb.run_suffix}'
        return project_name, run_name

    def set_config(self, config) -> None:
        final_list = []
        self._return_tuple_list(config,final_list)
        wandb.config.update(dict(final_list),allow_val_change=True)

    def _return_tuple_list(self, d: dict, final_list:list, parent_name=""):
        for k,v in d.items():
            if type(v) is DictConfig:
                parent_name_ = '.'.join([parent_name,k]) if parent_name!="" else k
                self._return_tuple_list(v,final_list,parent_name=parent_name_)
            else:
                v_ = ('.'.join([parent_name,k]) if parent_name!="" else k, v)
                final_list.append(v_)
    
    def save_object(self, object, fileName) -> None:
        if not os.path.isdir(os.path.join(wandb.run.dir, 'objects')):
            os.mkdir(os.path.join(wandb.run.dir, 'objects'))
        json.dump(object,open(os.path.join(wandb.run.dir, "objects" ,f"{fileName}.json"),'w'))

    def save_model(self, model, name, round) -> None:
        if not os.path.isdir(os.path.join(wandb.run.dir, 'models')):
            os.mkdir(os.path.join(wandb.run.dir, 'models'))
        torch.save({'weight':model.state_dict(),'round':round},\
            os.path.join(wandb.run.dir, "models" ,f"{name}.pt"))

    def add_scalar(self, tag, value, global_step=None) -> None:
        if global_step is None:
            wandb.log({tag:value})
        else:
            wandb.log({tag:value}, step=global_step)
    
    def add_summary_value(self, name, value):
        wandb.run.summary[name] = value
    
    def add_table(self, data, columns, title):
        self.add_scalar(title,wandb.Table(data=data, columns=columns),0)

    def add_local_var(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if name in self.local_store.keys():
                self.local_store[name]+=value
        else:
            self.local_store[name]=value


def exit_on_signal(sig, ret_code=0):
    signal.signal(sig, lambda *args: sys.exit(ret_code))


def shuffled_copy(x):
    x_copy = copy.copy(x)  # shallow copy
    np.random.shuffle(x_copy)
    return x_copy


def select_random_subset(x, portion: float):
    input_len = len(x)
    # drop at least one item but not all of them
    to_drop_num = max(1, min(input_len-1, math.ceil(input_len * portion)))
    to_drop_indexes = np.random.randint(0, input_len, to_drop_num)
    return np.delete(x, to_drop_indexes)


class MeasureMeter:
    def __init__(self, num_classes: int):
        self.__num_classes = num_classes
        self.__tp = torch.zeros(num_classes)
        self.__tn = torch.zeros(num_classes)
        self.__fp = torch.zeros(num_classes)
        self.__fn = torch.zeros(num_classes)
        self.__total = torch.zeros(num_classes)  # helper, it is just tp+tn+fp+fn

    @property
    def num_classes(self):
        return self.__num_classes

    def reset(self):
        self.__tp.fill_(0)
        self.__tn.fill_(0)
        self.__fp.fill_(0)
        self.__fn.fill_(0)
        self.__total.fill_(0)

    @property
    def accuracy_overall(self) -> float:
        return 100. * torch.sum(self.__tp) / torch.sum(self.__total)

    @property
    def accuracy_per_class(self) -> torch.Tensor:
        return 100. * torch.divide(self.__tp, self.__total)

    def update(self, predicted_batch: torch.Tensor, label_batch: torch.Tensor):
        for predicted, label in zip(predicted_batch, label_batch.view_as(predicted_batch)):
            # implement only accuracy
            if predicted.item() == label.item():
                self.__tp[label.item()] += 1
            self.__total[label.item()] += 1
