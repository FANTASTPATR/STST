import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.utils import expand_as_pair
import dgl.function as fn
import dgl.ops as ops

# import pickle


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout, num_heads):
        super(GAT, self).__init__()
        self._num_heads = num_heads
        self._out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        self._gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=F.relu) for _ in range(num_layers))
        self.dropout = dropout

    def forward(self, features, graph):
        x = features
        out = []
        for idx, layer in enumerate(self._gat_layers):
            x = layer(graph, x)
            x = torch.cat([x[:, i, :] for i in range(self._num_heads)], dim=1)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self._out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, feat_drop=0., attn_drop=0., neg_slope=0.1, activation=None):
        super(GATLayer, self).__init__()
        self._num_heads = num_heads
        self._in_src_dim, self._in_dst_dim = expand_as_pair(in_dim)
        self._out_dim = out_dim
        self._fc = nn.Linear(self._in_src_dim, out_dim * num_heads, bias=False)
        self._attn_left = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, out_dim)))
        self._attn_right = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, out_dim)))
        self._feat_drop = nn.Dropout(feat_drop)
        self._attn_drop = nn.Dropout(attn_drop)
        self._leaky_relu = nn.LeakyReLU(neg_slope)
        self._reset_parameters()
        self._activation = activation

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self._fc.weight, gain=gain)
        nn.init.xavier_uniform_(self._attn_right, gain=gain)
        nn.init.xavier_uniform_(self._attn_left, gain=gain)

    def forward(self, graph, feats):
        '''

        :param graph:
        :param feats: [batch_size*snp*num_node,hidden_dim]
        :return:
        '''
        # with open("./graph.pkl", "wb") as f:
        #     pickle.dump(graph, f)
        with graph.local_scope():
            h_src = self._feat_drop(feats)  # h_src:[batch_size*snp*num_node,hidden_dim]
            feat_src = feat_dst = self._fc(h_src).view(-1, self._num_heads,
                                                       self._out_dim)  # feat_src:[batch_size*snp*num_node,head,out_dim]
            el = (feat_src * self._attn_left).sum(dim=-1).unsqueeze(-1)  # el:[batch_size*snp*num_node,head,1]
            er = (feat_dst * self._attn_right).sum(dim=-1).unsqueeze(-1)  # er:[batch_size*snp*num_node,head,1]

            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            graph.apply_edges(fn.u_add_v(lhs_field='el', rhs_field='er', out='e'))
            e = self._leaky_relu(graph.edata.pop('e'))  # [edge_num,num_head,1]

            w = graph.edata['w'].unsqueeze(-1).unsqueeze(-1)
            w = torch.repeat_interleave(w, self._num_heads, 1)

            e = w * e
            graph.edata['a'] = self._attn_drop(ops.edge_softmax(graph, e))
            # with open("./attention_bi.pkl", "wb") as f:
            #     pickle.dump(graph.edata['a'], f)
            graph.update_all(fn.u_mul_e(lhs_field='ft', rhs_field='a', out='m'), fn.sum(msg='m', out='ft'))
            rst = graph.dstdata['ft']
            if self._activation:
                rst = self._activation(rst)
        return rst
