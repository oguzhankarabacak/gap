import torch
import dgl
import torch.optim as optim
from model import GAPModel
from utils import combined_loss, evaluate

def main():
    num_nodes = 100
    num_edges = 300
    num_features = 10
    num_partitions = 3
    hidden_features = 16

    graph = dgl.rand_graph(num_nodes, num_edges)
    features = torch.randn(num_nodes, num_features)

    model = GAPModel(num_features, hidden_features, num_partitions)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        partition_probs = model(graph, features)
        loss = combined_loss(graph, partition_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, loss: {loss.item()}')

    print('Training complete!')

    test_graph = dgl.rand_graph(num_nodes, num_edges)
    test_features = torch.randn(num_nodes, num_features)

    model.eval()
    with torch.no_grad():
        test_partition_probs = model(test_graph, test_features)

    test_cut_loss, test_balance_error = evaluate(test_graph, test_partition_probs)
    print(f"Test Normalized Cut Loss: {test_cut_loss}")
    print(f"Test Balance Error: {test_balance_error}")

if __name__ == '__main__':
    main()