import torch
import torchvision.models as models
from torchviz import make_dot
import networkx as nx
import dgl
import pydot

def convert_to_dgl(graph_dot):
    # Parse the DOT string with pydot
    pydot_graph = pydot.graph_from_dot_data(graph_dot.source)[0]
    # Convert pydot graph to networkx graph
    nx_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)
    # Convert networkx graph to DGL graph
    dgl_graph = dgl.from_networkx(nx_graph)
    return dgl_graph

def resnet_graph_dataset():
    resnet18 = models.resnet18()
    x = torch.randn(1, 3, 224, 224)
    y = resnet18(x)
    dot = make_dot(y, params=dict(resnet18.named_parameters()))
    dgl_graph = convert_to_dgl(dot)
    num_features = 10
    num_nodes = dgl_graph.number_of_nodes()
    features = torch.randn(num_nodes, num_features)
    dgl_graph.ndata['feat'] = features
    return dgl_graph