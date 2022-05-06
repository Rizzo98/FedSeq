import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class EmnistLocalDataset(Dataset):
    def __init__(self, images, labels, num_classes, client_id=-1, train=True, **kvargs):
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.client_id = client_id
        self.train = train


    def __getitem__(self, index):
        img = self.images[index]
        img = torch.tensor(img.reshape(-1, 28, 28),dtype=torch.float32)
        target = torch.tensor(self.labels[index])
        return img, target

    def __len__(self):
        return len(self.images)

    def get_subset_eq_distr(self, n: int):
        train_sorted_index = np.argsort(self.labels)
        train_img = self.images[train_sorted_index]
        train_label = self.labels[train_sorted_index]
        img_per_class = n//self.num_classes
        elements,count = np.unique(train_label,return_counts=True)
        assert all([c>img_per_class for c in count]), 'Too many exemplars!'
        range_index_per_class = [(sum(count[:i]),sum(count[:i])+count[i]) for i in range(len(elements))]
        subset_indexes = []
        for i in range(self.num_classes):
            subset_indexes += list(np.random.choice(np.arange(*range_index_per_class[i]), img_per_class, replace=False))
        
        subset_images = train_img[subset_indexes]
        subset_labels = train_label[subset_indexes]

        self.images = np.delete(train_img,subset_indexes, axis=0)
        self.labels = np.delete(train_label,subset_indexes)
        return EmnistLocalDataset(subset_images, subset_labels, self.num_classes)