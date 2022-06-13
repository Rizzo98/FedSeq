from torch.utils.data import Dataset
import torch
import numpy as np

class ShakespeareLocalDataset(Dataset):
    def __init__(self, x, y, num_classes=80, client_id=-1, **kvargs) -> None:
        self.client_id = client_id
        self.x = x
        self.labels = [y_[0] for y_ in y]
        self.num_classes = num_classes
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)
    
    def get_subset_eq_distr(self, n: int):
        sub_indexes = np.random.choice(list(range(len(self))),n, replace=False)
        sub_x = []
        sub_y = []
        for i in sub_indexes:
            sub_x.append(self.x[i])
            sub_y.append([self.labels[i]])
        self.x = [x for i,x in enumerate(self.x) if i not in sub_indexes]
        self.labels = [y for i,y in enumerate(self.labels) if i not in sub_indexes]
        return ShakespeareLocalDataset(sub_x, sub_y)