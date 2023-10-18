import torch
import torch_geometric

import matplotlib as plt
import numpy as np
import networkx
from torch_geometric.data import Data
from sklearn.manifold import TSNE


from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj


def loadDataset(name):
    dataset = TUDataset(root="dataset", name=name)
    if name != 'MUTAG':
        max_degree = 0
        degs = []
        for data in dataset:
            deg = torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
            degs.extend(deg.numpy())
            max_degree = max(max_degree, max(deg).item())
        # assign to one hot degree for each data (OneHotDegree receive maximum degree parameter)
        dataset.transform = torch_geometric.transforms.OneHotDegree(int(max_degree))
    
    return dataset

def toComplementary(g):
    c = abs(to_dense_adj(g.edge_index) - 1) - torch.eye(len(g.x))
    c = c[0].nonzero().t().contiguous()
    return c

def createComplementPairGraph(dataset):
    dataset_c = []
    for graph in dataset:
        edge_c = toComplementary(graph)
        dataset_c.append(Data(edge_index=edge_c, x=graph.x, y=graph.y))
    return dataset, dataset_c

def train_test_split(dataset_o, dataset_c, ratio, batch_size):
    total = len(dataset_o)
    # original graph
    g_train = dataset_o[:round(ratio*total)]
    g_test = dataset_o[round(ratio*total):]

    # complementary graph
    gc_train = dataset_c[:round(ratio*total)]
    gc_test = dataset_c[round(ratio*total):]
    
    # batch size
    seed = 12345
    g_train_loader = DataLoader(g_train, batch_size=batch_size, shuffle=False)
    g_test_loader = DataLoader(g_test, batch_size=batch_size, shuffle=False)

    gc_train_loader = DataLoader(gc_train, batch_size=batch_size, shuffle=False)
    gc_test_loader = DataLoader(gc_test, batch_size=batch_size, shuffle=False)
    
    return g_train_loader, g_test_loader, gc_train_loader, gc_test_loader
    
# print("dataset")

