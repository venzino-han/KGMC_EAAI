"""build graph with edge features"""
import numpy as np
import pandas as pd
import torch as th

import dgl 

import pandas as pd
import numpy as np
import scipy.sparse as sp

#######################
# Build graph
#######################

class UserItemGraph(object):
    """
    Build user-item graph for training 
    Bulid Homogeneous graph with df
    only user user-item pairs in range when extracting subgraph
    """
    def __init__(self, label_col:str, user_col:str, item_col:str, text_col:str,
                 df:pd.DataFrame, edge_idx_range:tuple,
                 ):
        df = df.copy()
        df['etype'] = df[label_col]

        self.user_col = user_col
        self.item_col = item_col
        self.text_col = text_col

        self._num_user = len(df[user_col].unique())
        self._num_item = len(df[item_col].unique())
        self._num_label = len(df[label_col].unique())
        
        # vid = vid + max(uid) + 1   
        df[item_col] += self._num_user
        u_idx, i_idx, = df[user_col].to_numpy(), df[item_col].to_numpy(), 
        etypes = df[label_col].to_numpy()
        labels = (df[label_col].to_numpy() - 1)/4
        ts = df['ts'].to_numpy()

        # use whole data to build main graph
        # add bidirect edges
        num_nodes = self._num_user + self._num_item
        src_nodes = np.concatenate((u_idx, i_idx))
        dst_nodes = np.concatenate((i_idx, u_idx))
        labels = np.concatenate((labels, labels))
        etypes = np.concatenate((etypes, etypes))
        ts = np.concatenate((ts, ts))

        print('df len ', len(df))
        print('nodes ', num_nodes)
        print('pairs ', src_nodes.shape, dst_nodes.shape )
        print('max id ', max(src_nodes), max(dst_nodes) )

        sp_mat = sp.coo_matrix((labels,(src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
        self.graph =dgl.from_scipy(sp_mat=sp_mat, idtype=th.int32)

        self.graph.ndata['node_id'] = th.tensor(list(range(num_nodes)), dtype=th.int32)

        self.graph.edata['original_src_idx'] = th.tensor(src_nodes, dtype=th.int32)
        self.graph.edata['original_dst_idx'] = th.tensor(dst_nodes, dtype=th.int32)
        self.graph.edata['label'] = th.tensor(labels, dtype=th.float32)
        self.graph.edata['etype'] = th.tensor(etypes, dtype=th.int32)
        self.graph.edata['ts'] = th.tensor(ts, dtype=th.int32)

        #extract subgraph pair idx
        start, end = edge_idx_range
        self.user_indices = th.tensor(u_idx[start:end], dtype=th.int32)
        self.item_indices = th.tensor(i_idx[start:end], dtype=th.int32)
        self.labels = th.tensor(labels[start:end], dtype=th.float32)

        self.user_item_pairs = self.get_user_item_pairs()

    def get_user_item_pairs(self):
        pairs = []
        for u, i in zip(self.user_indices, self.item_indices):
            pairs.append((u,i))
        return pairs
