import torch
import torch_geometric
import numpy as np

from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

# Previous proposed model
class CLComplement(torch.nn.Module):
    # merging type: o --> complement only, s --> substraction, c --> concatenation
    def __init__(self, dataset, hidden_channels, encoder, merging_type='o'):
        super(CLComplement, self).__init__()
        
        self.merging_type = merging_type
        
        # weight seed
        torch.manual_seed(42)
        self.conv1_o = encoder(dataset.num_node_features, hidden_channels)
        self.conv2_o = encoder(hidden_channels, hidden_channels)
        
        self.conv1_c = encoder(dataset.num_node_features, hidden_channels)
        self.conv2_c = encoder(hidden_channels, hidden_channels)
        
        # classification layer
        
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x_o, x_c, edge_index_o, edge_index_c, batch_o):
        x_o = self.conv1_o(x_o, edge_index_o)
        x_o = x_o.relu()
        x_o = self.conv2_o(x_o, edge_index_o)
        
        x_c = self.conv1_c(x_c, edge_index_c)
        x_c = x_c.relu()
        x_c = self.conv2_c(x_c, edge_index_c)

        # print(x_c.size())
        if (self.merging_type == 'o'):
            h = x_c
        elif (self.merging_type == 's'):
            h = x_o - x_c
        elif (self.merging_type == 'c'):
            h = torch.cat((x_o, x_c), 0)
        # print(h.size())

        
        h = global_add_pool(h, batch_o)
        
        h = self.lin(h)
        
        return h

# Proposed method
class ComplementarySupCon(torch.nn.Module):
    # merging type: o --> complement only, s --> substraction, c --> concatenation
    def __init__(self, dataset, hidden_channels, encoder, merging_type='o'):
        super(ComplementarySupCon, self).__init__()
        
        self.merging_type = merging_type
        
        # weight seed
        torch.manual_seed(42)
        self.conv1_o = encoder(dataset.num_node_features, hidden_channels)
        self.conv2_o = encoder(hidden_channels, hidden_channels)
        
        self.conv1_c = encoder(dataset.num_node_features, hidden_channels)
        self.conv2_c = encoder(hidden_channels, hidden_channels)
        
        # classification layer
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x_o, x_c, edge_index_o, edge_index_c, batch_o, classification = False):
        x_o = self.conv1_o(x_o, edge_index_o)
        x_o = x_o.relu()
        x_o = self.conv2_o(x_o, edge_index_o)

        x_c = self.conv1_c(x_c, edge_index_c)
        x_c = x_c.relu()
        x_c = self.conv2_c(x_c, edge_index_c)
        
        # print(x_c.size())
        if (self.merging_type == 'o'):
            h = x_c
        elif (self.merging_type == 's'):
            h = x_o - x_c
        elif (self.merging_type == 'c'):
            h = torch.cat((x_o, x_c), 0)
        
        # print(h.size())
        
        h = global_add_pool(h, batch_o)
        h = self.lin1(h)
        
        if (classification):
            h.relu()
            y = self.lin2(y)
            
            return h, y
            
        return h, x_o, x_c
    
class BaseModel(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, encoder):
        super(BaseModel, self).__init__()
        
        # weight seed
        torch.manual_seed(42)
        self.conv1 = encoder(dataset.num_node_features, hidden_channels)
        self.conv2 = encoder(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes) # for final classification

    def forward(self, x, edge_index, batch):
        # step 1. get node embedding using GCNConv layer
        x = self.conv1(x, edge_index)
        x = x.relu() # apply relu activation after conv
        x = self.conv2(x, edge_index)

        # step 2. add readout layer to aggregate all node features of graph
        e = global_add_pool(x, batch)

        # apply classifier (using linear)
        x = self.lin(e)

        return x, e