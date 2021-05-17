import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


class Aggregator(nn.Module):
    def __init__(self, hidden_dim, num_node):
        super(Aggregator, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(num_node, 1)

    def forward(self, inputs):
        '''

        :param inputs: [B,T,N,D]
        [B,T,N,D] ,[T,N,D]->[B,1,N,D]
        :return: [B,1,N,D]
        '''
        q = F.tanh(self.W_q(inputs))
        k = F.tanh(self.W_k(inputs)).transpose(-1, -2)
        attn = torch.einsum("...nd,...bc->...nc", q, k)
        attn = self.fc(attn)
        attn = F.softmax(attn, dim=1)
        # with open("./agg_attn_bi.pkl", "wb") as f:
        #     pickle.dump(attn, f)
        ret = torch.einsum("bsnd,bsnl->blnd", inputs, attn)
        return ret
