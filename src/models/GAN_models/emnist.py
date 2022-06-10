import torch
import torch.nn as nn
import torch.nn.functional as F

class EmnistGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(EmnistGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 3, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            EmnistGenerator.normal_init(self._modules[m], mean, std)

    def forward(self, input):
        return self.main(input)

class EmnistDiscriminator(nn.Module):
    def __init__(self, original_model, num_classes=62) -> None:
        super(EmnistDiscriminator,self).__init__()
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.binary_classifier = nn.Linear(num_classes,1)
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        #self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64*12*12, 128)
        #self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.binary_classifier = nn.Linear(num_classes,1)
        self.__set_weights_from_original(original_model)

    def __set_weights_from_original(self, original_model : nn.Module):
        state_dict = self.state_dict()
        for k in state_dict.keys():
            if 'model.'+k in original_model.state_dict().keys():
                state_dict[k] = original_model.state_dict()['model.'+k]
        self.load_state_dict(state_dict)

    def forward(self,x):
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,0.2,inplace=True)

        x = self.conv2(x)   
        x = self.bn2(x)     
        x = F.leaky_relu(x,0.2,inplace=True)

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x),0.2,inplace=True)
        x = F.leaky_relu(self.fc2(x),0.2,inplace=True)
        x = self.binary_classifier(x)
        return torch.sigmoid(x).view(-1)
        '''
        x = self.conv1(x)
        x = F.leaky_relu(x,0.2,inplace=True)
        x = self.conv2(x)
        x = F.leaky_relu(x,0.2,inplace=True)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x),0.2,inplace=True)
        x = F.leaky_relu(self.fc2(x),0.2,inplace=True)
        x = self.binary_classifier(x)
        return torch.sigmoid(x).view(-1)
