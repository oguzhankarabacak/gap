import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = torch.relu(h)
        h = self.conv2(graph, h)
        h = torch.relu(h)
        return h

class PartitioningModule(nn.Module):
    def __init__(self, in_feats, num_partitions):
        super(PartitioningModule, self).__init__()
        self.fc = nn.Linear(in_feats, num_partitions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embeddings):
        logits = self.fc(embeddings)
        partition_probs = self.softmax(logits)
        return partition_probs

class GAPModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_partitions):
        super(GAPModel, self).__init__()
        self.graph_sage = GraphSAGE(in_feats, h_feats)
        self.partitioning_module = PartitioningModule(h_feats, num_partitions)

    def forward(self, graph, features):
        embeddings = self.graph_sage(graph, features)
        partition_probs = self.partitioning_module(embeddings)
        return partition_probs