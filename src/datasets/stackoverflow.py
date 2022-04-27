from torch.utils.data import Dataset
import torch
import numpy as np

class StackoverflowLocalDataset(Dataset):
    def __init__(self, x, y, num_classes, device, client_id=-1) -> None:
        self.client_id = client_id
        self.x = np.array(x)
        self.labels = np.array(y)
        self.num_classes = num_classes
        self.device = device
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)
    
    def get_subset_eq_distr(self, n: int):
        if self.device == 'cpu':
            n = int(n*(1/34))

        sub_indexes = np.random.choice(list(range(len(self))),n, replace=False)
        sub_x = []
        sub_y = []
        for i in sub_indexes:
            sub_x.append(self.x[i])
            sub_y.append([self.labels[i]])
        self.x = [x for i,x in enumerate(self.x) if i not in sub_indexes]
        self.labels = [y for i,y in enumerate(self.labels) if i not in sub_indexes]

        return StackoverflowLocalDataset(sub_x, sub_y,self.num_classes,self.device)