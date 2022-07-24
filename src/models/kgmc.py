"""IGMC modules"""

import math 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv, NNConv, EGATConv

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class KGMC(nn.Module):
    # NNConv + RGCN convolution 
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=5, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, 
                multiply_by=1):
        super(KGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.egat_conv = EGATConv(in_node_feats=in_feats, in_edge_feats=1, 
                                  out_node_feats=in_feats, out_edge_feats=1,
                                  num_heads=2, 
                                 )
        self.convs = th.nn.ModuleList()
        self.convs.append(gconv(in_feats, latent_dim[1], num_relations, num_bases=num_bases, self_loop=True,))
        for i in range(1, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases=num_bases, self_loop=True,))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim[1:]), 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    # @profile
    def forward(self, subg, keyword_subg):
        # print(subg)
        # print(keyword_subg)
        subg = edge_drop(subg, self.edge_dropout, self.training)

        concat_states = []
        x = subg.ndata['x'].type(th.float32)
        kfeats = subg.edata.get('keywords', None)
        if kfeats is not None:
            x, _ = self.egat_conv(keyword_subg, x, efeats=kfeats,)
            x = th.tahn(x)
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(subg, x, subg.edata['etype'], 
                             norm=subg.edata['edge_mask'].unsqueeze(1)))
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)
        
        users = subg.ndata['nlabel'][:, 0] == 1
        items = subg.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            assert False
            # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph