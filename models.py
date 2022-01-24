import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
import timm
from torch_geometric.data import Data
import numpy as np

from layers import GCN, HGPSLPool
from utils.build_graph import BuildGraph


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_nodes = 16
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb
        self.fc_channels = 1280*7*7

        self.pretrain = timm.create_model('efficientnet_b0', pretrained=True)
        self.cnn = nn.Sequential(*list(self.pretrain.children())[:-2])

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin0 = torch.nn.Linear(self.fc_channels, self.num_features)
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def build_graph(self, data):
        source = np.array([], dtype=np.int)
        destination = np.array([], dtype=np.int)

        for graph_id in range(self.args.batch_size):
            start_index = graph_id * self.num_nodes
            build_graph = BuildGraph(self.num_nodes, start_index)
            [s, d] = build_graph.font_dense_graph()
            source = np.append(source, s)
            destination = np.append(destination, d)

        source = torch.tensor(source)
        destination = torch.tensor(destination)
        
        source = source.view(-1).tolist()[:-1]
        destination = destination.view(-1).tolist()[:-1]

        edge_index = torch.tensor([source, destination], dtype=torch.long).to(self.args.device)
        x = data
        batch = torch.tensor([[i]*self.num_nodes for i in range(self.args.batch_size)])
        batch = batch.view(-1).to(self.args.device)
        return x, edge_index, batch

    def forward(self, data):
        # extract feature from data
        data = data.permute(0, 2, 1, 3, 4)
        data = data.flatten(start_dim=0, end_dim=1)
        data = self.cnn(data)
        data = data.view(data.size()[0], -1)
        data = self.lin0(data)

        # convert to geometric Data structure
        x, edge_index, batch = self.build_graph(data)
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
