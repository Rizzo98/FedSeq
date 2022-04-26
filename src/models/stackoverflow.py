import torch.nn as nn

class SoverflowRnn(nn.Module):
    def __init__(self, num_classes, input_length=20, embed_size=96,hidden_size=670):
        super(SoverflowRnn,self).__init__()
        self.embedding = nn.Embedding(input_length,embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1)
        self.linear1 = nn.Linear(hidden_size,embed_size)
        self.linear2 = nn.Linear(embed_size, num_classes)

    def forward(self,x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, (h_, c_) = self.lstm(x)
        hidden = output[-1,:,:]
        x = self.linear1(hidden)
        x = self.linear2(x)
        return x