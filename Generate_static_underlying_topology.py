import numpy as np
import networkx as nx 

N = 1000
m = 3
k = m * 2


def f_node_neibour(graph):  
    number_record=np.zeros(N,dtype=int)
    neibour_record=np.zeros((N,N),dtype=int)-1
    for col_label,row_label in graph.edges():
        neibour_record[col_label,number_record[col_label]]=row_label
        neibour_record[row_label,number_record[row_label]]=col_label
        number_record[col_label]+=1
        number_record[row_label]+=1
    return neibour_record, number_record

def f_graph_tree_strc(graph, root):
    neibour_record, number_record = f_node_neibour(graph)
    max_val = max(number_record)
    active_node_list = []
    active_node_list.append(root)
    generation_node_record = []
    generation_edge_record = []
    generation_node_record.append([root])
    direct_neibour_record = np.zeros((N, max_val), dtype=np.int8) - 1
    direct_number_record = np.zeros(N, dtype=np.int8)
    tik = 0
    while len(active_node_list) < N:
        node_list = []
        edge_list = []
        seq = generation_node_record[tik]
        for vertex in seq:
            neibour = neibour_record[vertex, :number_record[vertex]]
            for id_x in neibour:
                if (id_x in active_node_list) == False:
                    node_list.append(id_x)
                    edge_list.append((vertex, id_x))
                    active_node_list.append(id_x)
                    direct_neibour_record[vertex, direct_number_record[vertex]] = id_x
                    direct_number_record[vertex] += 1
        generation_node_record.append(node_list)
        generation_edge_record.extend(edge_list)
        tik += 1
    tree_graph = nx.empty_graph()
    tree_graph.add_edges_from(generation_edge_record)
    return generation_node_record, direct_neibour_record, \
           direct_number_record, tree_graph
           
def graph_to_matrix(graph):
    static_adj_mat = np.zeros((N, N), dtype=np.int16)
    for x, y in graph.edges():
        static_adj_mat[x, y] = 1
        static_adj_mat[y, x] = 1
    return static_adj_mat
    
graph = nx.random_graphs.barabasi_albert_graph(N, m)
root = np.random.choice(range(N))
generation_node_record, direct_neibour_record, \
            direct_number_record, tree_graph = f_graph_tree_strc(graph, root)

static_adj_mat_origin = graph_to_matrix(graph)
static_adj_mat_tree = graph_to_matrix(tree_graph)

str1 = 'static_adj_mat_origin' + '_' + str(N) + '_' + str(k) + '.npy'
str2 = 'static_adj_mat_tree' + '_' + str(N) + '_' + str(k) + '.npy'

np.save(str1, static_adj_mat_origin)
np.save(str2, static_adj_mat_tree)

print(nx.algorithms.tree.recognition.is_tree(tree_graph))
print(nx.average_clustering(graph))
