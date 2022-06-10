# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import torch.utils.model_zoo as model_zoo

import torchvision.models.resnet as resnet
import torch
import torch.nn as nn

from .task2vec import ProbeNetwork

_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn


class ResNet(resnet.ResNet, ProbeNetwork):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        # Saves the ordered list of layers. We need this to forward from an arbitrary intermediate layer.
        self.layers = [
            self.conv1, self.bn1, self.relu,
            self.maxpool, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
            lambda z: torch.flatten(z, 1), self.fc
        ]

    @property
    def classifier(self):
        return self.fc

    # @ProbeNetwork.classifier.setter
    # def classifier(self, val):
    #     self.fc = val

    # Modified forward method that allows to start feeding the cached activations from an intermediate
    # layer of the network
    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x

class RNN(ProbeNetwork):
    def __init__(self, num_classes, embed_size=8, hidden_size=100):
        super(RNN,self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1)
        self.fc = nn.Linear(hidden_size,num_classes)
        self.layers = [
            self.embedding, self.lstm, self.fc
        ]

    def forward(self,x,start_from=0):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, (h_, c_) = self.lstm(x)
        last_hidden = output[-1,:,:]
        x = self.fc(last_hidden)
        return x

    @property
    def classifier(self):
        return self.fc

    # @ProbeNetwork.classifier.setter
    # def classifier(self, val):
    #     self.fc = val

    # Modified forward method that allows to start feeding the cached activations from an intermediate
    # layer of the network
    '''
    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return '''


@_add_model
def resnet18(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model: ProbeNetwork = ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(resnet.model_urls['resnet18'])
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    model.train()
    model.in_channels = 3
    return eval_resnet_bn_layers(model)

@_add_model
def resnet34(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    model.train()
    model.in_channels = 3
    return eval_resnet_bn_layers(model)

@_add_model
def rnn(pretrained=False, num_classes=1000):
    """Constructs a simple rnn model.
    Args:
        pretrained (bool): not implemented
    """
    model: ProbeNetwork = RNN(num_classes=num_classes)
    model.train()
    model.in_channels = None
    return model
    
def get_model(model_name, pretrained=False, num_classes=1000):
    try:
        return _MODELS[model_name](pretrained=pretrained, num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Architecture {model_name} not implemented.")

def eval_resnet_bn_layers(model):
    for layer in model.layers:
        if not hasattr(layer, '__name__') or layer.__name__!='<lambda>': #to skip lambda functions
            if layer._get_name() == 'BatchNorm2d':
                layer.eval()
            elif layer._get_name() == 'Sequential':
                for l in layer:
                    l.bn1.eval()
                    l.bn2.eval()
                    if l.downsample is not None:
                        for b in l.downsample:
                            if b._get_name() == 'BatchNorm2d':
                                b.eval()
    return model