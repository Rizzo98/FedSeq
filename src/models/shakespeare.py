import torch.nn as nn

class ShakespeareRnn(nn.Module):
    def __init__(self, num_classes, input_length=80, embed_size=8,hidden_size=100):
        super(ShakespeareRnn,self).__init__()
        self.embedding = nn.Embedding(input_length,embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=2)
        self.linear = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, (h_, c_) = self.lstm(x)
        last_hidden = output[-1,:,:]
        x = self.linear(last_hidden)
        return x