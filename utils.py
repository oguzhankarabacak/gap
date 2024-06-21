import dgl
import torch
def extract_node_degrees_from_dgl_graph(graph):
    node_degrees = graph.in_degrees() + graph.out_degrees()
    return node_degrees.float()

def create_adjacency_matrix(graph):
    return graph.adjacency_matrix().to_dense()

def modified_normalized_cut_loss(graph, partition_probs, epsilon=1e-5):
    '''
    gamma = partition_probs(Transpose) * node degrees
    normalized_cut_loss = (partition_probs /(element wise) gamma) * ( 1 - partition_probs)(transpose) * (element_wise) adjacency_matrix

    :param graph:
    :param partition_probs:
    :return:
    '''
    gamma = partition_probs.T @ extract_node_degrees_from_dgl_graph(graph)
    term1 = torch.div(partition_probs, gamma + epsilon)
    term2 = (1 - partition_probs).T
    term3 = create_adjacency_matrix(graph)
    loss = (term1 @ term2) * term3
    return loss.sum()

def modified_normalized_cut_loss_2(graph, partition_probs):
    node_degrees = extract_node_degrees_from_dgl_graph(graph)
    adjacency_matrix = create_adjacency_matrix(graph)
    gamma = partition_probs.T @ node_degrees
    term1 = torch.div(partition_probs, gamma)
    term2 = (1 - partition_probs).T
    term1_term2 = term1 @ term2
    loss_matrix = term1_term2 * adjacency_matrix
    loss = loss_matrix.sum()
    return loss

def modified_balanced_partition_error(partition_probs):
    '''
    reduce-sum(1.T * partition_probs - n/g)^2
    :param partition_probs: nxg tensor
    :param n: given the number of nodes in the graph
    :param g: number of partitions
    :return:
    '''
    n = partition_probs.size(0)
    g = partition_probs.size(1)
    ones_vector = torch.ones(n, 1)
    balance_error = (((ones_vector.T @ partition_probs) - (n / g)) ** 2).sum()
    #print((ones_vector.T @ partition_probs), "Shape: " ,(ones_vector.T @ partition_probs).shape)
    return balance_error

def modified_loss(graph, partition_probs, alpha=1e-5, epoch=None):
    '''

    :param graph:
    :param partition_probs:
    :return:
    '''
    cut_loss = modified_normalized_cut_loss(graph, partition_probs)
    balance_error = modified_balanced_partition_error(partition_probs)
    if epoch is not None and epoch % 10 == 0:
        print(f'Epoch: {epoch}, Cut Loss: {cut_loss}, Balance Error: {alpha*balance_error}, Total Loss: {cut_loss + alpha*balance_error}')
    return cut_loss + alpha*balance_error

# Evaluation Metrics
def edge_cut_ratio(graph, partition_probs):
    with graph.local_scope():
        graph.ndata['p'] = partition_probs
        graph.apply_edges(dgl.function.u_mul_v('p', 'p', 'e')) #
        edge_cut = graph.edata['e'].sum().item()
    total_edges = graph.num_edges()
    return edge_cut / total_edges

def balancedness(partition_probs, n, g):
    partition_sizes = partition_probs.sum(dim=0)
    ideal_size = n/g
    balance_error = ((partition_sizes - ideal_size) ** 2).mean().item()
    balance_error = balance_error ** 0.5
    balance_error /= ideal_size
    return 1-balance_error

def evaluate(graph, partition_probs):
    n = graph.number_of_nodes()
    g = partition_probs.size(1)
    cut_ratio = edge_cut_ratio(graph, partition_probs)
    balance_score = balancedness(partition_probs, n, g)
    return cut_ratio, balance_score