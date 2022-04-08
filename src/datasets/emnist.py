import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from collections import Counter



class EmnistLocalDataset(Dataset):
    def __init__(self, images, labels, num_classes, client_id=-1, train=True):
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.client_id = client_id
        self.train = train
        self.transform = transforms.ToTensor()


    def __getitem__(self, index):
        img = self.images[index]
        img = self.transform(img).reshape(-1, 28, 28)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)

    def get_subset_eq_distr(self, n: int): # da fare
        train_sorted_index = np.argsort(self.labels)
        train_img = self.images[train_sorted_index]
        train_label = self.labels[train_sorted_index]
        num_img_per_class = len(self)//self.num_classes
        subset_indexes = []
        """for i in range(self.num_classes):
            subset_indexes += list(np.random.choice(num_img_per_class, img_per_class, replace=False) + num_img_per_class * i)
        """
        subset_images = train_img[subset_indexes]
        subset_labels = train_label[subset_indexes]

        self.images = np.delete(train_img,subset_indexes, axis=0)
        self.labels = np.delete(train_label,subset_indexes)
        return EmnistLocalDataset(subset_images, subset_labels, self.num_classes)