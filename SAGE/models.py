from SAGE.layers import SageConv
import torch.nn.functional as F
import torch.nn as nn

class Sage(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Sage, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x