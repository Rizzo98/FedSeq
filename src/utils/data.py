import numpy as np
from src.datasets import *
import torchvision
import random
import logging
import json
import os
import torch
from tqdm import tqdm
from src.utils import non_iid_partition_with_dirichlet_distribution

log = logging.getLogger(__name__)


def get_dataset(requested_dataset, **kwargs):
    dataset_getter = {"cifar10": get_CIFAR10_data,
                      "cifar100": get_CIFAR100_data,
                      "shakespeare_niid": get_shakespeare_data,
                      "shakespeare_iid": get_shakespeare_data,
                      "shakespeare_niid_full": get_shakespeare_data,
                      "emnist_niid": get_emnist_data,
                      "emnist_niid_subset": get_emnist_data,
                      "emnist_iid": get_emnist_data,
                      "soverflow_niid": get_soverflow_data,
                      "soverflow_iid": get_soverflow_data
                      }
    dataset_transformation = {"cifar10": cifar_transform,
                              "cifar100": cifar_transform,
                              "shakespeare_niid": shakespeare_transform,
                              "shakespeare_iid": shakespeare_transform,
                              "shakespeare_niid_full": shakespeare_transform,
                              "emnist_niid": emnist_transform,
                              "emnist_niid_subset": emnist_transform,
                              "emnist_iid": emnist_transform,
                              "soverflow_niid": soverflow_transform,
                              "soverflow_iid": soverflow_transform}

    if requested_dataset not in dataset_getter:
        raise KeyError(f"the requested dataset {requested_dataset} is not supported")
    return dataset_getter[requested_dataset], dataset_transformation[requested_dataset]


def get_CIFAR10_data(**kwargs):
    # wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    # tar -zxvf cifar-10-python.tar.gz
    CIFAR10train = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True)
    CIFAR10test = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True)

    return CIFAR10train.data, np.array(CIFAR10train.targets), CIFAR10test.data, np.array(
        CIFAR10test.targets)


def get_CIFAR100_data(**kwargs):
    #wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    #tar -zxvf cifar-100-python.tar.gz
    CIFAR100train = torchvision.datasets.CIFAR100(root="./datasets", train=True, download=True)
    CIFAR100test = torchvision.datasets.CIFAR100(root="./datasets", train=False, download=True)

    return CIFAR100train.data, np.array(CIFAR100train.targets), CIFAR100test.data, np.array(
        CIFAR100test.targets)

def cifar_transform(centralized, train_x, train_y, test_x, test_y, **kwargs):
    if centralized: #if centralized-> converting dataset to centralized
        return train_x, train_y, test_x, test_y
    else: #converting to federated
        dataset_size = kwargs['dataset_size']
        num_clients = kwargs['num_clients']
        alpha= kwargs['alpha']
        dataset_class = kwargs['dataset_class']
        dataset_num_class = kwargs['dataset_num_class']

        shard_size = dataset_size // num_clients
        max_iter = kwargs['params'].max_iter_dirichlet
        rebalance = kwargs['params'].rebalance
        shuffle_start_index = kwargs['params'].shuffle_start_index
        order_by_superclass = kwargs['params'].order_by_superclass if 'order_by_superclass' in kwargs['params'] else None

        if shard_size < 1:
            raise ValueError("shard_size should be at least 1")
        if alpha == 0:  # Non-IID
            local_datasets, test_datasets = create_non_iid(train_x, test_x, train_y, test_y, num_clients,
                                                        shard_size,
                                                        dataset_class, dataset_num_class, shuffle_start_index, order_by_superclass)
        else:
            local_datasets, test_datasets = create_using_dirichlet_distr(train_x, test_x, train_y, test_y,
                                                                        num_clients,
                                                                        alpha, max_iter, rebalance, shard_size, dataset_class,
                                                                        dataset_num_class)
        return local_datasets, test_datasets

def get_shakespeare_data(**kwargs):
    test_file_name = 'test_sampled.json'
    if kwargs['dataset_name']=='shakespeare_niid':
        train_data = json.load(open(os.path.join(os.getcwd(),'datasets','Shakespeare','train_sampled_niid.json')))
    elif kwargs['dataset_name']=='shakespeare_iid':
        train_data = json.load(open(os.path.join(os.getcwd(),'datasets','Shakespeare','train_sampled_iid.json')))
    elif kwargs['dataset_name']=='shakespeare_niid_full':
        train_data = json.load(open(os.path.join(os.getcwd(),'datasets','Shakespeare','train_full_niid.json')))
        test_file_name = 'test_processed.json'
    test_data = json.load(open(os.path.join(os.getcwd(),'datasets', 'Shakespeare', test_file_name)))
    return train_data['x'], train_data['y'], test_data['x'], test_data['y']

def shakespeare_transform(centralized, train_x, train_y, test_x, test_y, **kwargs):
    if centralized: #if centralized-> converting dataset to centralized
        transformed_train_x = []
        transformed_train_y = []
        for client in train_x:
            transformed_train_x += client
        for client in train_y:
            transformed_train_y += client
        return transformed_train_x, transformed_train_y, test_x, test_y
    else: #converting to federated
        dataset_class = kwargs['dataset_class']
        local_datasets = []
        for i,(x,y) in enumerate(zip(train_x,train_y)):
            local_datasets.append(dataset_class(x,y,client_id=i))
        test_dataset = dataset_class(test_x, test_y)
        return local_datasets, test_dataset

def get_emnist_data(**kwargs): 
    datasets_path = os.path.join(os.getcwd(),'datasets','EMNIST')
    if 'emnist_niid' in kwargs['dataset_name']:
        train_dir = os.path.join(datasets_path, 'train', 'EMNIST_niid')
    elif 'emnist_iid' in kwargs['dataset_name']:
        train_dir = os.path.join(datasets_path, 'train', 'EMNIST_iid')
    files = os.listdir(train_dir)
    files = [f for f in files if f.endswith('.json')]
    train_data = []
    train_labels = []
    if '_subset' in kwargs['dataset_name']:
        files = files[:4]
    for f in tqdm(files,desc='Loading training files'):
        training_dict = json.load(open(os.path.join(train_dir, f)))
        train_data += training_dict['x']
        train_labels += training_dict['y']
    test_dir = os.path.join(datasets_path, 'test')
    files = os.listdir(test_dir)
    files = [f for f in files if f.endswith('.json')]
    test_data = []
    test_labels = []
    #if '_subset' in kwargs['dataset_name']:
    #    files = files[:3]
    for f in tqdm(files,desc='Loading test files'):
        test_dict = json.load(open(os.path.join(test_dir, f)))
        test_data += test_dict['x']
        test_labels += test_dict['y']
    if '_subset' in kwargs['dataset_name']:
        train_data, train_labels = train_data[:368], train_labels[:368]
    #    test_data, test_labels = test_data[:8172], test_labels[:8172]
    return train_data, train_labels, test_data, test_labels

def emnist_transform(centralized, train_x, train_y, test_x, test_y, **kwargs):
    if centralized: #if centralized-> converting dataset to centralized
        transformed_train_x = []
        transformed_train_y = []
        for client in train_x:
            transformed_train_x += client
        for client in train_y:
            transformed_train_y += client
        return transformed_train_x, transformed_train_y, test_x, test_y
    else: #converting to federated
        dataset_class = kwargs['dataset_class']
        local_datasets = []
        for i,(x,y) in enumerate(zip(train_x,train_y)):
            local_datasets.append(dataset_class(x,y,kwargs['dataset_num_class'],client_id=i))
        test_dataset = dataset_class(test_x, test_y,kwargs['dataset_num_class'])
        return local_datasets, test_dataset

def get_soverflow_data(**kwargs):
    datasets_path = os.path.join(os.getcwd(),'datasets','stackoverflow')
    if kwargs['dataset_name']=='soverflow_niid':
        train_dir = os.path.join(datasets_path, 'train', 'soverflow_niid')
        #train_dir = os.path.join(datasets_path, 'train', 'soverflow_niid', 'textual_dataset')
    elif kwargs['dataset_name']=='soverflow_iid':
        train_dir = os.path.join(datasets_path, 'train', 'soverflow_iid')
    files = os.listdir(train_dir)
    files = [f for f in files if f.endswith('.json')]
    train_data = []
    train_labels = []
    if len(files) > 8:
        files = ['train_0.json', 'train_1.json', 'train_2.json', 'train_3.json']
    for f in tqdm(files,desc='Loading training files'):
        file = open(os.path.join(train_dir, f))
        training_dict = json.load(file)
        train_data += training_dict['x']
        train_labels += training_dict['y']
        file.close()
    #fatto per far corrispondere l'ordine dei due dataset di stackoverflow, si puo tranquillamente poi rimuovere
    ordered_data = train_data[30000:35000] + train_data[10000:15000] + train_data[35000:] + train_data[15000:20000] + train_data[25000:30000] + train_data[5000:10000] + train_data[20000:25000] + train_data[:5000]
    ordered_labels = train_labels[30000:35000] + train_labels[10000:15000] + train_labels[35000:] + train_labels[15000:20000] + train_labels[25000:30000] + train_labels[5000:10000] + train_labels[20000:25000] + train_labels[:5000]
    train_data = ordered_data
    train_labels = ordered_labels
    test_dir = os.path.join(datasets_path, 'test')
    files = os.listdir(test_dir)
    files = [f for f in files if f.endswith('.json')]
    test_data = []
    test_labels = []
    if kwargs['device'] == 'cpu':
        files = files[:4]
    for f in tqdm(files[:4],desc='Loading test files'):
        file = open(os.path.join(test_dir, f))
        test_dict = json.load(file)
        test_data += test_dict['x']
        test_labels += test_dict['y']
    return train_data, train_labels, test_data, test_labels

def soverflow_transform(centralized, train_x, train_y, test_x, test_y, **kwargs):
    if centralized: #if centralized-> converting dataset to centralized
        transformed_train_x = []
        transformed_train_y = []
        for client in train_x:
            transformed_train_x += client
        for client in train_y:
            transformed_train_y += client
        return transformed_train_x, transformed_train_y, test_x, test_y
    else: #converting to federated
        dataset_class = kwargs['dataset_class']
        local_datasets = []
        for i,(x,y) in enumerate(zip(train_x,train_y)):
            local_datasets.append(dataset_class(x,y, kwargs['dataset_num_class'], device = kwargs['device'], client_id=i))
        test_dataset = dataset_class(test_x, test_y, kwargs['dataset_num_class'], device = kwargs['device'], train=False)
        return local_datasets, test_dataset

def create_datasets(dataset, num_clients, alpha, **kwargs):
    
    dataset_getter, transform = get_dataset(dataset.name)
    dataset_class = eval(dataset.dataset_class)
    dataset_num_class = dataset.dataset_num_class
    dataset_size = dataset.dataset_size
    train_img, train_label, test_img, test_label = dataset_getter(**kwargs, dataset_name=dataset.name)

    local_datasets, test_datasets = transform(False, train_img, train_label, test_img, test_label,\
        num_clients=num_clients, alpha=alpha, dataset_class=dataset_class,\
            dataset_num_class=dataset_num_class, dataset_size=dataset_size, params=dataset.params, device=kwargs['device'])

    return local_datasets, test_datasets


def create_non_iid(train_img, test_img, train_label, test_label, num_clients, shard_size,
                   dataset_class, dataset_num_class, shuffle_start_index, order_by_superclass):
    
    if dataset_class == CifarLocalDataset and dataset_num_class == 100 and order_by_superclass:
        coarse_mappings = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
        train_coarse_label = np.array([coarse_mappings[y] for y in train_label])
        train_sorted_index = np.lexsort((train_label, train_coarse_label))
    else: 
        train_sorted_index = np.argsort(train_label)
    train_img = train_img[train_sorted_index]
    train_label = train_label[train_sorted_index]

    shard_start_index = [i for i in range(0, len(train_img), shard_size)]
    if shuffle_start_index:
        random.shuffle(shard_start_index)
    log.info(f"divide data into {len(shard_start_index)} shards of size {shard_size}")

    num_shards = len(shard_start_index) // num_clients
    local_datasets = []
    for client_id in range(num_clients):
        _index = num_shards * client_id
        img = np.concatenate([
            train_img[shard_start_index[_index +
                                        i]:shard_start_index[_index + i] +
                                           shard_size] for i in range(num_shards)
        ],
            axis=0)

        label = np.concatenate([
            train_label[shard_start_index[_index +
                                          i]:shard_start_index[_index +
                                                               i] +
                                             shard_size] for i in range(num_shards)
        ],
            axis=0)
        local_datasets.append(dataset_class(img, label, dataset_num_class))

    test_sorted_index = np.argsort(test_label)
    test_img = test_img[test_sorted_index]
    test_label = test_label[test_sorted_index]
    test_dataset = dataset_class(test_img, test_label, dataset_num_class, train=False)

    return local_datasets, test_dataset


def create_using_dirichlet_distr(train_img, test_img, train_label, test_label,
                                 num_clients, alpha, max_iter, rebalance, shard_size, dataset_class, dataset_num_class):
    d = non_iid_partition_with_dirichlet_distribution(
        np.array(train_label), num_clients, dataset_num_class, alpha, max_iter)

    if rebalance:
        storage = []
        for i in range(len(d)):
            if len(d[i]) > (shard_size):
                difference = round(len(d[i]) - (shard_size))
                toSwitch = np.random.choice(
                    d[i], difference, replace=False).tolist()
                storage += toSwitch
                d[i] = list(set(d[i]) - set(toSwitch))

        for i in range(len(d)):
            if len(d[i]) < (shard_size):
                difference = round((shard_size) - len(d[i]))
                toSwitch = np.random.choice(
                    storage, difference, replace=False).tolist()
                d[i] += toSwitch
                storage = list(set(storage) - set(toSwitch))

        for i in range(len(d)):
            if len(d[i]) != (shard_size):
                log.warning(f'There are some clients with more than {shard_size} images')

    # Lista contenente per ogni client un'istanza di Cifar10LocalDataset ->local_datasets[client]
    local_datasets = []
    for client_id in d.keys():
        # img = np.concatenate( [train_img[list_indexes_per_client_subset[client_id][c]] for c in range(n_classes)],axis=0)
        img = train_img[d[client_id]]
        # label = np.concatenate( [train_label[list_indexes_per_client_subset[client_id][c]] for c in range(n_classes)],axis=0)
        label = train_label[d[client_id]]
        local_datasets.append(dataset_class(img, label, dataset_num_class))

    test_sorted_index = np.argsort(test_label)
    test_img = test_img[test_sorted_index]
    test_label = test_label[test_sorted_index]

    test_dataset = dataset_class(test_img, test_label, dataset_num_class, train=False)

    return local_datasets, test_dataset

def dataset_from_dataloader(data_loader, intended_channels, extend_labels):
    data, labels = next(iter(data_loader))
    if data.ndim == 4: #image
        if data.shape[1] > intended_channels:
            raise ValueError(f'intended_channels must be >= than data_loader # of channels. Got {data_loader[0].shape[1]} > {intended_channels}.')
        elif data.shape[1] < intended_channels:
            data = np.repeat(data, intended_channels, axis=1)
            for i, (x, y) in enumerate(data_loader):
                if i!=0:
                    data = torch.vstack((data, 
                                np.repeat(x, intended_channels, axis=1)))
                    labels = torch.cat((labels, y))
            dataset = torch.utils.data.TensorDataset(data, labels)
            return dataset
    elif extend_labels:
        labels = torch.cat((data[:,1:], labels.reshape((-1,1))), dim=-1)
        for i, (x, y) in enumerate(data_loader):
            if i!=0:
                data = torch.vstack((data, x))
                labels = torch.vstack((labels, torch.cat((x[:,1:], y.reshape((-1,1))), dim=-1)))
        dataset = torch.utils.data.TensorDataset(data, labels)
        return dataset   
    dataset = data_loader.dataset
    return dataset