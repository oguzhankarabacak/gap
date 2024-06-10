import torch
import dgl
import torch.optim as optim
from model import GAPModel
from utils import combined_loss, evaluate, modified_loss
import networkx as nx
from dataset import resnet_graph_dataset

def generate_erdos_renyi_graph(num_nodes, prob):
    g = nx.erdos_renyi_graph(num_nodes, prob)
    return dgl.from_networkx(g)

def main():
    num_nodes = 1000
    num_features = 10
    num_partitions = 3
    hidden_features = 16

    graph = generate_erdos_renyi_graph(num_nodes, 0.1)
    features = torch.randn(num_nodes, num_features)

    #graph = resnet_graph_dataset()
    #features = graph.ndata['feat']


    model = GAPModel(num_features, hidden_features, num_partitions)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        partition_probs = model(graph, features)
        #loss = combined_loss(graph, partition_probs)
        loss = modified_loss(graph, partition_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, loss: {loss.item()}')

    print('Training complete!')

    # Generate a test graph
    test_graph = resnet_graph_dataset()
    test_features = test_graph.ndata['feat']

    model.eval()
    with torch.no_grad():
        test_partition_probs = model(test_graph, test_features)

    # Evaluate on the test graph
    test_cut_ratio, test_balance_score = evaluate(test_graph, test_partition_probs)
    print(f"Test Edge Cut Ratio: {test_cut_ratio}")
    print(f"Test Balancedness: {test_balance_score}")

if __name__ == '__main__':
    main()