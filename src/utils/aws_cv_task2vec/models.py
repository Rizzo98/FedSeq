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
from transformers import GPT2Tokenizer
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
    
    def has_batchnorm(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                return True
        return False

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


import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(ProbeNetwork):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        C.client_vocab_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.client_vocab_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert (type_given and not params_given) or (not type_given and params_given) # exactly one of these
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.client_vocab_size, bias=False)
        self.layers = [block for block in self.transformer.h] + [self.transformer.ln_f, self.lm_head] #we are gonna use all the model, so it's here just for compatibility
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        #print("number of parameters: %.2fM" % (n_params/1e6,))
        self.stoi = {} #empty, to be loaded in from the state dict
        self.itos = {}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        config.client_vocab_size = 50257
        model = GPT(config)
        model.apply(model._init_weights)
        sd = model.state_dict()
        sd = {k:v for k,v in sd.items() if 'lm_head' not in k}

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if (not k.endswith('attn.masked_bias') and 'lm_head' not in k)] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx, start_from=0):
        if start_from == 0:
            device = idx.device
            b, t = idx.size()
            assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

            # forward the GPT model itself
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = idx
        for layer in self.layers[start_from:]:
            x = layer(x)
        logits = x.view(-1, x.size(-1))
        return logits

    @property
    def classifier(self):
        return self.lm_head


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
    model.extend_labels = False
    model.classifier_opts = {}
    model.skip_layers = 0
    model.loader_opts = {}
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
    model.extend_labels = False
    model.classifier_opts = {}
    model.skip_layers = 0
    model.loader_opts = {}
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

@_add_model
def charGPT(pretrained=False, num_classes=1000):
    #7.46M parameters
    config = CN()
    # model
    config.model = GPT.get_default_config()
    config.model.model_type = 'gpt-mini'
    config.model.vocab_size = 24895
    config.model.block_size = 80
    config.model.client_vocab_size = num_classes
    model = GPT(config.model)
    if pretrained:
        PATH = './src/utils/aws_cv_task2vec/pretrained_models/chargpt_wikipedia_model.pt'
        pretrained = torch.load(PATH)
        state_dict = pretrained['weights']
        state_dict = {k: v for k, v in state_dict.items() if 'lm_head' not in k}
        model.itos = pretrained['itos']
        model.stoi = pretrained['stoi']
        model.load_state_dict(state_dict, strict=False)
    model.train()
    model.in_channels = None
    model.extend_labels = True
    model.classifier_opts = {'epochs':20}
    model.skip_layers = 0
    model.loader_opts = {}
    return model

@_add_model
def minGPT(pretrained=True, num_classes=1000):
    model = GPT.from_pretrained(model_type='gpt2')
    model.train()
    model.in_channels = None
    model.extend_labels = False
    model.classifier_opts = {'epochs':3}
    model.skip_layers = 10
    model.loader_opts = {'num_samples':1000}
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