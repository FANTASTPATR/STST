import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.utils import expand_as_pair
import dgl.function as fn


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout, num_channels):
        super(GCN, self).__init__()
        self._num_layers = num_layers
        self._out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        self._gcn_layers = nn.ModuleList(
            GCNLayer(in_dim=in_dim, out_dim=int(out_dim / num_channels), num_channels=num_channels, activation=F.relu)
            for
            _ in range(num_layers))
        self.dropout = dropout
        self._num_channels = num_channels

    def forward(self, features, graph):
        x = features
        out = []
        for idx, layer in enumerate(self._gcn_layers):
            x = layer(graph, x)
            x = torch.cat([x[:, i, :] for i in range(self._num_channels)], dim=1)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self._out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_channels, norm="both", activation=None):
        super(GCNLayer, self).__init__()
        self._fc = nn.Linear(in_dim, in_dim)
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._norm = norm
        self._num_channels = num_channels
        self._weight = nn.Parameter(torch.FloatTensor(size=(num_channels, out_dim, out_dim)))
        self._bias = nn.Parameter(torch.FloatTensor(size=(num_channels, out_dim)))
        self._reset_parameters()
        self._activation = activation

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self._weight.data, gain=gain)
        nn.init.xavier_uniform_(self._bias.data, gain=gain)

    def forward(self, graph, feats):
        '''

        :param graph:
        :param feats: [batch_size*snp*num_node,hid_dim]
        :return:
        '''
        with graph.local_scope():
            h_src = self._fc(feats)  # [batch_size*snp*num_node,hid_dim]
            feat_src = feat_dst = h_src.view(-1, self._num_channels,
                                             self._out_dim)  # [batch_size*snp*num_node,channels,out_dim]
            if self._norm == "both":
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
            if self._in_dim > self._out_dim:
                if self._weight is not None:
                    # feat_src:[b,c,d]
                    # weight:[c,d,d]
                    feat_src = torch.einsum("bcd,cde->bce", feat_src, self._weight)
                    # feat_src = torch.matmul(feat_src, self._weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.u_mul_e(lhs_field='h', rhs_field='w', out='m'), fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.u_mul_e(lhs_field='h', rhs_field='w', out='m'), fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if self._weight is not None:
                    rst = torch.einsum("bcd,cde->bce", feat_src, self._weight)

            if self._norm is not None:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self._bias is not None:
                rst = rst + self._bias

            if self._activation is not None:
                rst = self._activation(rst)
            return rst
