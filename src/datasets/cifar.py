import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.functional import one_hot
import numpy as np
import torch

class CifarLocalDataset(Dataset):
    def __init__(self, images, labels, num_classes, client_id=-1, train=True):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.client_id = client_id
        self.train = train
        if train:
            self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    def __getitem__(self, index):
        img = Image.fromarray(self.images[index], mode='RGB')
        img = self.transform(img)
        target = self.labels[index]
        #img, target = self.__instahideTransform(index,4)
        return img, target

    def __len__(self):
        return len(self.images)

    def __instahideTransform(self, index, klam):
        original = self.x_values[index]
        mixed_x = torch.zeros_like(original)
        mixed_y = torch.zeros((1,self.num_classes))

        for i in range(klam):
            x = self.x_values[self.cur_selects[index,i]].clone()
            ys_onehot = self.y_values[self.cur_selects[index,i]].clone()

            mixed_x += self.cur_lams[index,i] * x
            mixed_y += self.cur_lams[index,i] * ys_onehot
        return mixed_x, mixed_y
    
    def generate_mapping(self, klam:int, lower_bound:float, upper_bound:float, return_tensor=True):
        alpha = [1.0]*klam
        lams = np.random.dirichlet(alpha=alpha,
                                    size=len(self.images))

        selects = np.asarray([
            np.random.permutation(len(self.images))
            for _ in range(klam)
        ])
        selects = np.transpose(selects)

        for i in range(len(self.images)):
            # enforce that k images are non-repetitive
            while len(set(selects[i])) != klam:
                selects[i] = np.random.randint(0, len(self.images),klam)
            if klam > 1:
                while (lams[i].max() > upper_bound) or (lams[i].min() < lower_bound):  # upper bounds a single lambda
                    lams[i] = np.random.dirichlet(alpha=alpha)
        if return_tensor:
            self.cur_lams = torch.from_numpy(lams).float()
            self.cur_selects = torch.from_numpy(selects).long()
        else:
            self.cur_lams = np.asarray(lams)
            self.cur_selects = np.asarray(selects)
        
        self.x_values = torch.stack([self.__getitem__(i)[0] for i in range(len(self.images))])

        self.y_values = torch.from_numpy(np.asarray([self.__getitem__(i)[1] for i in range(len(self.images))]))
        self.y_values = one_hot(self.y_values, num_classes=self.num_classes).float()


    def get_subset_eq_distr(self, n: int):
        img_per_class = n//self.num_classes
        train_sorted_index = np.argsort(self.labels)
        train_img = self.images[train_sorted_index]
        train_label = self.labels[train_sorted_index]
        num_img_per_class = len(self)//self.num_classes
        subset_indexes = []
        for i in range(self.num_classes):
            subset_indexes += list(np.random.choice(num_img_per_class, img_per_class, replace=False) + num_img_per_class * i)
        
        subset_images = train_img[subset_indexes]
        subset_labels = train_label[subset_indexes]

        self.images = np.delete(train_img,subset_indexes, axis=0)
        self.labels = np.delete(train_label,subset_indexes)

        return CifarLocalDataset(subset_images, subset_labels, self.num_classes)
