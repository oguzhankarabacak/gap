import dgl

def normalized_cut_loss(graph, partition_probs):
    with graph.local_scope():
        graph.ndata['p'] = partition_probs
        graph.apply_edges(dgl.function.u_mul_v('p', 'p', 'e'))
        cut_loss = graph.edata['e'].sum()
    return cut_loss

def balance_loss(partition_probs, n, g):
    partition_sizes = partition_probs.sum(dim=0)
    balance_error = ((partition_sizes - (n / g)) ** 2).sum()
    return balance_error

def combined_loss(graph, partition_probs, alpha=1.0):
    n = graph.number_of_nodes()
    g = partition_probs.size(1)
    cut_loss = normalized_cut_loss(graph, partition_probs)
    balance_error = balance_loss(partition_probs, n, g)
    return cut_loss + alpha * balance_error


def evaluate(graph, partition_probs):
    n = graph.number_of_nodes()
    g = partition_probs.size(1)
    cut_loss = normalized_cut_loss(graph, partition_probs)
    balance_error = balance_loss(partition_probs, n, g)
    return cut_loss.item(), balance_error.item()