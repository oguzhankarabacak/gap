import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, hidden_size):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, h_feats, 'mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = torch.relu(h)
        h = self.conv2(graph, h)
        h = torch.relu(h)
        return h

class PartitioningModule(nn.Module):
    def __init__(self, in_feats,hidden_size, num_partitions):
        super(PartitioningModule, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_partitions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embeddings):
        x = F.relu(self.fc1(embeddings))
        logits = self.fc2(x)
        partition_probs = self.softmax(logits)
        return partition_probs

class GAPModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_partitions):
        super(GAPModel, self).__init__()
        self.graph_sage = GraphSAGE(in_feats, h_feats, 512)
        self.partitioning_module = PartitioningModule(h_feats,hidden_size=32 , num_partitions=num_partitions)

    def forward(self, graph, features):
        embeddings = self.graph_sage(graph, features)
        partition_probs = self.partitioning_module(embeddings)
        return partition_probs

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)  # Initialize biases to a small positive constant
