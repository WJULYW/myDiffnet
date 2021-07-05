'''
    author: Jiyao WANG
    e-mail: jiyaowang130@gmail.com
    release date:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class diffnet():
    def __init__(self,conf):
        self.conf=conf
        self.supply_set=(
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX'
        )

    def startConstructGraph(self):


    def inputSupply(self,data_dict):
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        self.social_neighbors_dense_shape=np.array([self.conf.num_users,self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape=np.array([self.conf.num_users,self.conf.num_items]).astype(np.int64)

        self.social_neighbors_sparse_matrix=torch
