import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from scipy import sparse as sp

from GAT import GAT
from aggregator import Aggregator
# from GCN import GCN

import pickle


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # graph_adj = sp.csr_matrix(args.adj_mx)
        # graph = dgl.from_scipy(graph_adj, eweight_name='w')
        self._args = args
        self._ckpnt = self._generate_checkpoints(self._args.snp_len, self._args.seq_in)
        self._graphs = self._generate_batch_graphs()
        self._linear_in = nn.Linear(self._args.in_dim, self._args.hidden_dim)
        self._mp_model = GAT(in_dim=self._args.hidden_dim, out_dim=self._args.hidden_dim,
                             num_layers=self._args.conv_layers,
                             dropout=self._args.dropout, num_heads=self._args.num_heads)

        # self._mp_model = GCN(in_dim=self._args.hidden_dim, out_dim=self._args.hidden_dim,
        #                      num_layers=self._args.conv_layers,
        #                      dropout=self._args.dropout, num_channels=self._args.num_heads)
        self._aggregator = Aggregator(num_node=self._args.num_node, hidden_dim=self._args.hidden_dim)
        self._layer_norm = nn.LayerNorm([1, self._args.num_node, self._args.hidden_dim], elementwise_affine=False)
        self._spatial_emb = nn.Parameter(torch.zeros(self._args.num_node, self._args.hidden_dim), requires_grad=True)
        self._temporal_emb = nn.Parameter(torch.zeros(self._args.num_node, self._ckpnt[-1]), requires_grad=True)
        self._reset_parameters()
        if self._args.meta_decode:
            self._construct_p1 = nn.Linear(self._args.hidden_dim, 1)
            self._construct_p2 = nn.Linear(self._args.num_node, self._args.seq_out)
            self._construct_p3 = nn.Linear(self._args.batch_size, 1)
            self._construct_p4 = nn.Linear(self._args.seq_out, self._args.hidden_dim)
        else:
            self._linear_out_1 = nn.Linear(1, self._args.seq_out)
            self._linear_out_2 = nn.Linear(self._args.hidden_dim, 1)

    def _reset_parameters(self):
        # gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self._spatial_emb.data, gain=1.414)
        nn.init.xavier_uniform_(self._temporal_emb.data, gain=1.414)

    def forward(self, inputs):
        # inputs:[B,H,N,D]

        if inputs.shape[1] != self._ckpnt[-1]:
            pad_len = self._ckpnt[-1] - inputs.shape[1]
            inputs = inputs.transpose(1, 3)
            inputs = F.pad(inputs, (pad_len, 0, 0, 0))
            inputs = inputs.transpose(1, 3)  # inputs:[B,H,N,D]
        x = self._linear_in(inputs)
        x = x + self._spatial_emb
        x = x.transpose(1, -1)
        x = x + self._temporal_emb
        x = x.transpose(1, -1)
        last_res = None
        LEFT = 0
        for idx, RIGHT in enumerate(self._ckpnt):
            features_slice = x[:, LEFT:RIGHT, :, :].contiguous()
            if idx != 0:
                features_slice = torch.cat([last_res, features_slice], dim=1)
            residual = features_slice
            features_slice = features_slice.view(self._args.batch_size * self._args.num_node * self._args.snp_len,
                                                 self._args.hidden_dim).contiguous()
            ret = self._mp_model(features_slice, graph=self._graphs)
            last_res = ret.view(self._args.batch_size, self._args.snp_len, self._args.num_node,
                                self._args.hidden_dim).contiguous()
            last_res = last_res + residual
            last_res = self._aggregator(last_res)
            last_res = self._layer_norm(last_res)
            LEFT = RIGHT

        if self._args.meta_decode:
            # [B,1,N,D]->[B,P,N,1]
            # w1:[B,1,N,D]->[1,P]
            w1 = F.tanh(self._construct_p1(last_res).squeeze(-1))  # [B,1,N]
            w1 = self._construct_p2(w1)  # [B,1,P]
            w1 = self._construct_p3(w1.transpose(-1, 0)).squeeze(-1).transpose(0, 1)  # [1,P]
            out = F.relu(torch.matmul(last_res.transpose(-1, 1), w1)).transpose(-1, 1)  # [B,P,N,D]
            # [B,P,N,D] ->[B,P,N,1]
            # w2:[1,P]->[D,1]
            w2 = self._construct_p4(w1).transpose(0, 1)  # [D,1]
            out = torch.matmul(out, w2)


        else:
            out = F.relu(self._linear_out_1(last_res.transpose(-1, 1)))
            out = self._linear_out_2(out.transpose(-1, 1))
        return out

    def _generate_batch_graphs(self):
        '''
        Generate batched dgl.Graphs
        :return: dgl.Graph(not two step) (dgl.Graph,dgl.Graph)(two step)
        '''
        # Heterogeneous Spatial Temporal Graph
        hstg = self._generate_HSTG(self._args.adj_mx, self._args.snp_len, self._args.connect_type,
                                   self._args.uni_direction, self._args.dense_conn, self._args.sync)

        batch_hstg = dgl.batch([hstg for _ in range(self._args.batch_size)])
        return batch_hstg.to(self._args.device)

    @staticmethod
    def _generate_checkpoints(snp_len, seq_in):
        '''
        generate indexes of snapshots that generate groups
        :param snp_len: the number of snapshots to form a group
        :param seq_in: the length of input sequence
        :return: list
        '''
        mark = snp_len
        checkpoints = []
        while True:
            checkpoints.append(mark)
            if mark != seq_in:
                mark += snp_len - 1
            else:
                break
            if mark >= seq_in:
                checkpoints.append(mark)
                break
        return checkpoints

    @staticmethod
    def _generate_HSTG(adj_mx, snp_len, connect_type, uni_direction, dense_connection, sync):
        '''
        Generate the Heterogeneous Spatial Temporal Graph
        :param adj_mx: adjacency matrix
        :param snp_len: the number of snapshots to form a group
        :param connect_type: the type of connection of different snapshots
        :return: dgl.graph
        '''

        def generate_HSTG_adj_mx_uni_direction():
            def minmax(A):
                return (A - torch.min(A)) / (torch.max(A) - torch.min(A))

            if not sync:
                ret_adj = torch.zeros(adj_mx.shape[0] * snp_len, adj_mx.shape[0] * snp_len)
                for idx in range(snp_len):
                    ret_adj[idx * adj_mx.shape[0]:idx * adj_mx.shape[1] + adj_mx.shape[0],
                    idx * adj_mx.shape[1]:idx * adj_mx.shape[1] + adj_mx.shape[1]] = adj_mx
                # with open("./adj_mx.pkl", "wb") as f:
                #     pickle.dump(ret_adj, f)
                return ret_adj
            else:
                if connect_type == "adj" or connect_type == "identity":
                    if connect_type == "adj":
                        connecnt_mx = adj_mx
                    else:
                        connecnt_mx = torch.eye(adj_mx.shape[0])
                    ret_adj = torch.zeros(adj_mx.shape[0] * snp_len, adj_mx.shape[0] * snp_len)
                    for idx in range(snp_len):
                        if idx != snp_len - 1:
                            if dense_connection:
                                temporal_connections = torch.cat([connecnt_mx for _ in range(snp_len - 1 - idx)],
                                                                 dim=-1)
                                st_connections = torch.cat([adj_mx, temporal_connections], dim=-1)
                            else:
                                st_connections = torch.cat([adj_mx, connecnt_mx], dim=-1)
                        else:
                            st_connections = adj_mx

                        ret_adj[idx * adj_mx.shape[0]:idx * adj_mx.shape[1] + st_connections.shape[0],
                        idx * adj_mx.shape[1]:idx * adj_mx.shape[1] + st_connections.shape[1]] = st_connections
                    # with open("./adj_mx.pkl", "wb") as f:
                    #     pickle.dump(ret_adj, f)
                    return ret_adj
                elif connect_type == "power":
                    ret_adj = torch.zeros(adj_mx.shape[0] * snp_len, adj_mx.shape[0] * snp_len)
                    for idx in range(snp_len):
                        if idx != snp_len - 1:
                            temporal_connections = torch.cat(
                                [minmax(torch.matrix_power(adj_mx, power + 1)) for power in range(snp_len - idx - 1)],
                                dim=-1)
                            st_connections = torch.cat([adj_mx, temporal_connections], dim=-1)
                        else:
                            st_connections = adj_mx
                        ret_adj[idx * adj_mx.shape[0]:idx * adj_mx.shape[1] + st_connections.shape[0],
                        idx * adj_mx.shape[1]:idx * adj_mx.shape[1] + st_connections.shape[1]] = st_connections
                    # with open("./adj_mx.pkl", "wb") as f:
                    #     pickle.dump(ret_adj, f)
                    return ret_adj

        def generate_HSTG_adj_mx_bi_direction():
            if connect_type == "adj" or connect_type == "identity":
                if connect_type == "adj":
                    connecnt_mx = adj_mx
                else:
                    connecnt_mx = torch.eye(adj_mx.shape[0])
                ret_adj = torch.zeros(adj_mx.shape[0] * snp_len, adj_mx.shape[0] * snp_len)
                for idx in range(snp_len):
                    st_connections = torch.cat([connecnt_mx for _ in range(snp_len)], dim=-1)
                    ret_adj[idx * adj_mx.shape[0]:idx * adj_mx.shape[0] + st_connections.shape[0],
                    0:idx * adj_mx.shape[0] + st_connections.shape[1]] = st_connections
                return ret_adj

        if uni_direction:
            hstg_adj = generate_HSTG_adj_mx_uni_direction()
        else:
            hstg_adj = generate_HSTG_adj_mx_bi_direction()
        hstg_adj = sp.csr_matrix(hstg_adj)
        graph = dgl.from_scipy(hstg_adj, eweight_name='w')
        return graph
