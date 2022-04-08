import copy
import math
import pickle
from functools import wraps
import logging
import os
import random
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
    def __init__(self, dataset_name, algo, entity="fedseq-thesis", device=None) -> None:
        project_name, run_name = self._generate_project_run_name(dataset_name, algo, device)
        wandb.init(project=project_name, entity=entity, config ={})
        wandb.run.name = run_name
        self.local_store = dict()
    
    def _generate_project_run_name(self, dataset_name, algo, device=None):
        project_name = ""
        iid_ness = ""
        if "_" in dataset_name:
            project_name = dataset_name.split('_')[0]
            if algo.type!='centralized':
                iid_ness = f'_split:{dataset_name.split("_")[1]}'
        else:
            project_name = dataset_name
            if 'common' in algo.params:
                iid_ness = f'_alpha:{algo.params.common.alpha}'
        run_name = f'{algo.type}{iid_ness}'
        if device=='cpu':
            project_name = 'test-project'
        return project_name, run_name

    def set_config(self, config) -> None:
        final_list = []
        self._return_tuple_list(config,final_list)
        wandb.config.update(dict(final_list))

    def _return_tuple_list(self, d: dict, final_list:list, parent_name=""):
        for k,v in d.items():
            if type(v) is DictConfig:
                parent_name_ = '.'.join([parent_name,k]) if parent_name!="" else k
                self._return_tuple_list(v,final_list,parent_name=parent_name_)
            else:
                v_ = ('.'.join([parent_name,k]) if parent_name!="" else k, v)
                final_list.append(v_)

    def save_model(self, model, name='model') -> None:
        if not os.path.isdir(os.path.join(wandb.run.dir, 'models')):
            os.mkdir(os.path.join(wandb.run.dir, 'models'))
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "models" ,f"{name}.pt"))

    def add_scalar(self, tag, value, global_step=None) -> None:
        if global_step is None:
            wandb.log({tag:value})
        else:
            wandb.log({tag:value}, step=global_step)
    
    def add_summary_value(self, name, value):
        wandb.run.summary[name] = value
    
    def add_table(self, data, columns, title):
        self.add_scalar(title,wandb.Table(data=data, columns=columns))

    def add_local_var(self, name, value):
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
