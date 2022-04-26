from torch.utils.data import Dataset
import torch
import numpy as np

class StackoverflowLocalDataset(Dataset):
    def __init__(self, x, y, client_id=-1, **kvargs) -> None:
        self.client_id = client_id
        self.x = x
        self.labels = [y_[0] for y_ in y]
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)
    
    def get_subset_eq_distr(self, n: int):
        sorted_index = np.argsort(self.labels)
        x = self.x[sorted_index]
        y = self.labels[sorted_index]
        x_per_class = n//self.num_classes
        elements, count = np.unique(y, return_counts=True)
        assert all([c>x_per_class for c in count]), 'Too many exemplars!'
        range_index_per_class = [(sum(count[:i]),sum(count[:i])+count[i]) for i in range(len(elements))]
        subset_indexes = []
        for i in range(self.num_classes):
            subset_indexes += list(np.random.choice(np.arange(*range_index_per_class[i]), x_per_class, replace=False))
        
        sub_x = x[subset_indexes]
        sub_y = y[subset_indexes]

        self.x = np.delete(x,subset_indexes, axis=0)
        self.labels = np.delete(y,subset_indexes)
        return StackoverflowLocalDataset(sub_x, sub_y)