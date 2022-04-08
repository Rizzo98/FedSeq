import torch.nn as nn
import torch.nn.functional as func


class emnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(emnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x