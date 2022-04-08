from matplotlib.pyplot import step
import torch
from src.datasets import CifarLocalDataset
from src.utils import MeasureMeter
from src.losses import CrossEntropyLoss
from src.algo import Algo
from torch.utils.data import DataLoader
import numpy as np
import wandb
import os

def test_on_data(config, model, do_train, device, writer):
    d = {
        'Cifar10C': test_Cifar10C
    }
    assert do_train and config.model==None or not do_train and config.model!=None, \
        'If a pre-trained model is not passed, do_train must be True!'
    if config.model!=None:
        run_path = '/'.join(config.model.split('/')[:3])
        model_name = '/'.join(config.model.split('/')[3:])
        state_dict = torch.load(wandb.restore(model_name, run_path=run_path).name)
        model.load_state_dict(state_dict)
    assert config.dataset in d.keys(), 'Dataset not implemented!'
    writer.set_config({
        'Dataset':config.dataset,
        'Model': config.model if config.model!=None else 'Trained this run'
        })
    d[config.dataset](model, writer, device)

def test_Cifar10C(model, writer, device='cpu'):
    path = './datasets/CIFAR-10-C'
    labels = np.load(os.path.join(path,'labels.npy'))
    labels = labels.astype(np.int_)
    accuracies_per_transform = dict()
    column_names = ['Transformation','Severity_1','Severity_2','Severity_3','Severity_4','Severity_5', 'Overall']
    data_list = []
    for name in os.listdir(path):
        if name!='labels.npy':
            f = os.path.join(path,name)
            raw_data = np.load(f)
            row = [name[:-4]]
            for severity,index in enumerate(range(0,50000,10000)):
                data = CifarLocalDataset(raw_data[index:index+10000],labels[index:index+10000],10,train=False)
                dataloader = DataLoader(data, batch_size=64)
                meter = MeasureMeter(10)
                Algo.test(model,meter,device,CrossEntropyLoss(),dataloader)
                accuracies_per_transform[name[:-4]] = meter.accuracy_overall
                print(f'Transformation:{name[:-4]}, Severity: {severity+1}, Accuracy: {meter.accuracy_overall}')
                row.append(meter.accuracy_overall)
            row.append(np.mean(row[1:]))
            data_list.append(row)
        writer.add_table(data_list,column_names,'Cifar10C results')