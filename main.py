import torch
import dgl
import torch.optim as optim
from model import GAPModel, weights_init
from utils import evaluate, modified_loss
import networkx as nx


def generate_erdos_renyi_graph(num_nodes, prob):
    g = nx.erdos_renyi_graph(num_nodes, prob, directed=False)
    return dgl.from_networkx(g)


def train(model, graphs=[], features=[], optimizer=None, num_epochs=1000):
    for epoch in range(num_epochs):
        for graph,index in zip(graphs,range(len(graphs))):
            model.train()
            partition_probs = model(graph, features[index])
            loss = modified_loss(graph, partition_probs, epoch = epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('Training complete!')
    for graph, index in zip(graphs, range(len(graphs))):
        with torch.no_grad():
            train_partition_probs = model(graph, features[index])
        test_cut_ratio, test_balance_score = evaluate(graph, train_partition_probs)
        print(f"Train Edge Cut Ratio: {test_cut_ratio}")
        print(f"Train Balancedness: {test_balance_score}")

    return model

def main():
    #random seed
    torch.manual_seed(0)
    num_nodes = 1000
    num_features = 1518
    num_partitions = 3
    hidden_features = 64
    graph_number = 1

    graphs = []
    features = []
    for i in range(graph_number):
        graph = generate_erdos_renyi_graph(num_nodes, 0.01)
        feature = torch.randn(num_nodes, num_features)
        graph.ndata['feat'] = feature
        features.append(feature)
        graphs.append(graph)

    model = GAPModel(num_features, hidden_features, num_partitions)

    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=7.5e-5, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    num_epochs = 1000

    model = train(model, graphs, features, optimizer, num_epochs)

    print('Test Results')
    # Generate a test graph
    test_num_nodes = 10000
    test_num_features = 1518
    test_graph = generate_erdos_renyi_graph(test_num_nodes, 0.1)
    test_features = torch.randn(test_num_nodes, test_num_features)


    model.eval()
    with torch.no_grad():
        test_partition_probs = model(test_graph, test_features)

    # Evaluate on the test graph
    test_cut_ratio, test_balance_score = evaluate(test_graph, test_partition_probs)
    print(f"Test Edge Cut Ratio: {test_cut_ratio}")
    print(f"Test Balancedness: {test_balance_score}")

if __name__ == '__main__':
    main()