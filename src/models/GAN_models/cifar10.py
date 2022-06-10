import torch
import torch.nn as nn
import torch.nn.functional as F

class Cifar10Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Cifar10Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            Cifar10Generator.normal_init(self._modules[m], mean, std)

    def forward(self, input):
        return self.main(input)

class Cifar10Discriminator(nn.Module):
    def __init__(self, original_model, num_classes=10) -> None:
        super(Cifar10Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)
        self.binary_classifier = nn.Linear(num_classes,1)
        self.__set_weights_from_original(original_model)

    def __set_weights_from_original(self, original_model : nn.Module):
        state_dict = self.state_dict()
        for k in state_dict.keys():
            if 'model.'+k in original_model.state_dict().keys():
                state_dict[k] = original_model.state_dict()['model.'+k]
        self.load_state_dict(state_dict)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,0.2,inplace=True)

        x = self.conv2(x)   
        x = self.bn2(x)     
        x = F.leaky_relu(x,0.2,inplace=True)

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x),0.2,inplace=True)
        x = F.leaky_relu(self.fc2(x),0.2,inplace=True)
        x = self.fc3(x)
        x = self.binary_classifier(x)
        return torch.sigmoid(x).view(-1)
