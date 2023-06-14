import networkx as nx 
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from math import exp

N = 1000
tol_time = 1000
n_net = 40
num_of_edge = N - 1
k = 6
alpha1, alpha2 = 2.5, 2.0

def matrix_to_graph(matrix):
    num, num = matrix.shape
    graph = nx.empty_graph()
    for x in range(num):
        for y in range(x, num):
            if matrix[x, y] > 0:
                graph.add_edge(x, y)
    return graph

def f_node_neibour(graph):  
    number_record = np.zeros(N, dtype=np.int16)
    neibour_record = np.zeros((N, N), dtype=np.int16)-1
    for col_label, row_label in graph.edges():
        neibour_record[col_label, number_record[col_label]] = row_label
        neibour_record[row_label, number_record[row_label]] = col_label
        number_record[col_label] += 1
        number_record[row_label] += 1
    return neibour_record, number_record

def f_generate_generation(graph, root):
    neibour_record, number_record = f_node_neibour(graph)
    max_val = max(number_record)
    active_node_list = []
    active_node_list.append(root)
    generation_node_record = []
    generation_node_record.append([root])
    direct_neibour_record = np.zeros((N, max_val), dtype=np.int32) - 1
    direct_number_record = np.zeros(N, dtype=np.int32)
    tik = 0
    while len(active_node_list) < N:
        node_list = []
        for vertex in generation_node_record[tik]:
            neibour = neibour_record[vertex, :number_record[vertex]]
            for id_x in neibour:
                if (id_x in active_node_list) == False:
                    node_list.append(id_x)
                    active_node_list.append(id_x)
                    direct_neibour_record[vertex, direct_number_record[vertex]] = id_x
                    direct_number_record[vertex] += 1
        generation_node_record.append(node_list)
        tik += 1
    return generation_node_record, direct_neibour_record, \
           direct_number_record

def f_node_generation(generation_node_record, generation_num):
    generation_num_record = np.zeros(generation_num, dtype=np.int16)
    generation_node_record_array = np.zeros((generation_num, N), dtype=np.int16) - 1
    for g in range(generation_num):
        seq1 = generation_node_record[g]
        length = len(seq1)
        generation_num_record[g] = length
        generation_node_record_array[g, :length] = seq1 
    return generation_node_record_array, generation_num_record


@njit
def f_generate_cond_dist(alpha, length):
    cond_prob_vec = np.zeros(length)
    for i in range(length):
        cond_prob_vec[i] = 1 - exp(-alpha)
    return cond_prob_vec 

@njit
def f_cal_prob(trajectory, cond_prob_vec, time):
    m = 0
    for i in range(time):
        if trajectory[i] == 1:
            m = i
    index = time - m - 1
    val = cond_prob_vec[index]
    return val 

@njit
def f_cal_dist(val_x, val_y, val_z):
    prob_array = np.zeros(4)
    prob_array[0] = val_z
    prob_array[1] = val_x - val_z
    prob_array[2] = val_y - val_z
    prob_array[3] = 1 + val_z - val_x - val_y
    return prob_array

@njit
def f_find_edge_index(all_edge_array, num_of_edge, seq):
    for i in range(num_of_edge):
        temp = all_edge_array[i, :]
        if np.prod(temp == seq):
            return i
    return -1

@njit
def f_single_turn(cond_prob_vec_node, cond_prob_vec_edge, 
                  generation_node_record, generation_num_record,
                  direct_neibour_record, direct_number_record,
                  all_edge_array, num_of_edge, generation_num):
    trajectory_node = np.zeros((N, tol_time+1), dtype=np.int8)
    for i in range(N):
        trajectory_node[i, 0] = 1
        
    trajectory_edge = np.zeros((num_of_edge, tol_time+1), dtype=np.int8)
    for i in range(num_of_edge):
        trajectory_edge[i, 0] = 1
        
    for time in range(1, tol_time+1):
        res_array = np.zeros(N, dtype=np.int32) - 1
        for g in range(generation_num-1):
            root_array = generation_node_record[g, :generation_num_record[g]]
            for root in root_array:
                val_root = f_cal_prob(trajectory_node[root, :], cond_prob_vec_node, time)
                if g == 0:
                    rand = np.random.random()
                    if rand < val_root:
                        res_root = 1
                        res_array[root] = 1
                    else:
                        res_root = 0
                        res_array[root] = 0
                else:
                    res_root = res_array[root]
                neibour = direct_neibour_record[root, : direct_number_record[root]]
                for vertex in neibour:
                    if vertex > root:
                        x, y = root, vertex
                    else:
                        x, y = vertex, root
                    seq = np.array([x, y])
                    edge_index = f_find_edge_index(all_edge_array, num_of_edge, seq)
                    val_leaf = f_cal_prob(trajectory_node[vertex, :], cond_prob_vec_node, time)
                    val_edge = f_cal_prob(trajectory_edge[edge_index, :], cond_prob_vec_edge, time)
                    prob_array = f_cal_dist(val_root, val_leaf, val_edge)
                    if np.sum(prob_array < 0) > 0:
                        print(time)
                        print(prob_array)
                        print('Distribution incompatibility!')
                        break
                    if res_root == 1:
                        rand = np.random.random()
                        val = prob_array[0] / val_root
                        if rand < val:
                            res_array[vertex] = 1
                            trajectory_edge[edge_index, time] = 1
                        else:
                            res_array[vertex] = 0
                            trajectory_edge[edge_index, time] = 0
                    else:
                        rand = np.random.random()
                        val = prob_array[2] / (1 - val_root)
                        if rand < val:
                            res_array[vertex] = 1
                            trajectory_edge[edge_index, time] = 0
                        else:
                            res_array[vertex] = 0
                            trajectory_edge[edge_index, time] = 0
        for j in range(N):
            trajectory_node[j, time] = res_array[j]
    return trajectory_node, trajectory_edge

        

def main():
    root = 0
    str1 = 'static_adj_mat_origin' + '_' + str(N) + '_' + str(k) + '.npy'
    str2 = 'static_adj_mat_tree' + '_' + str(N) + '_' + str(k) + '.npy'
    matrix = np.load(str2)
    graph = matrix_to_graph(matrix)
    generation_node_record, direct_neibour_record, direct_number_record = \
                                            f_generate_generation(graph, root)
    generation_num = len(generation_node_record)
    generation_node_record, generation_num_record = \
                     f_node_generation(generation_node_record, generation_num)         
    all_edge_list = graph.edges()
    num_of_edge = len(all_edge_list)
    all_edge_array = np.zeros((num_of_edge, 2), dtype=np.int16)
    tik = 0
    for x, y in all_edge_list:
        if x > y:
            x, y = y, x
        all_edge_array[tik, 0] = x
        all_edge_array[tik, 1] = y
        tik += 1
    alpha = alpha1
    cutoff = tol_time * 10
    cond_prob_vec_node = f_generate_cond_dist(alpha, cutoff)
    
    alpha = alpha2
    cond_prob_vec_edge = f_generate_cond_dist(alpha, cutoff)
    pool = ProcessPoolExecutor()
    cond_prob_vec_node = [cond_prob_vec_node] * n_net
    cond_prob_vec_edge = [cond_prob_vec_edge] * n_net
    generation_node_record = [generation_node_record] * n_net
    generation_num_record = [generation_num_record] * n_net
    direct_neibour_record = [direct_neibour_record] * n_net
    direct_number_record = [direct_number_record] * n_net
    all_edge_array = [all_edge_array] * n_net
    num_of_edge = [num_of_edge] * n_net
    generation_num = [generation_num] * n_net
    result_list = list(pool.map(f_single_turn, cond_prob_vec_node, cond_prob_vec_edge, 
                  generation_node_record, generation_num_record,
                  direct_neibour_record, direct_number_record,
                  all_edge_array, num_of_edge, generation_num))
    net_trajectory_node = np.zeros((n_net, N, tol_time+1), dtype=np.int8)
    for i in range(n_net):
        net_trajectory_node[i, :, :] = result_list[i][0]

    matrix = np.load(str1)
    graph = matrix_to_graph(matrix)
    all_edge_list = graph.edges()
    num_of_edge_origin = len(all_edge_list)
    all_edge_array = np.zeros((num_of_edge_origin, 2), dtype=np.int16)
    tik = 0
    for x, y in all_edge_list:
        if x > y:
            x, y = y, x
        all_edge_array[tik, 0] = x
        all_edge_array[tik, 1] = y
        tik += 1
    net_trajectory_edge = f_orgin_net_trajectory(net_trajectory_node, 
                                                  all_edge_array,
                                                  num_of_edge_origin)
    return net_trajectory_node, net_trajectory_edge

@njit
def f_orgin_net_trajectory(net_trajectory_node, all_edge_array, num_of_edge):
    net_trajectory_edge = np.zeros((n_net, num_of_edge, tol_time+1), dtype=np.int8)
    for i in range(n_net):
        for time in range(tol_time+1):
            tik = 0
            for j in range(num_of_edge):
                id_x = all_edge_array[j, 0]
                id_y = all_edge_array[j, 1]
                if time == 0:
                    net_trajectory_edge[i, tik, time] = 1
                else:
                    net_trajectory_edge[i, tik, time] = \
                                 net_trajectory_node[i, id_x, time] *  \
                                 net_trajectory_node[i, id_y, time]
                tik += 1
    return net_trajectory_edge
    
if __name__=="__main__":
    num1, num2 = N, k
    net_trajectory_node, net_trajectory_edge = main()
    str1 = 'net_trajectory_node' + '_' + str(num1) + '_' + str(num2) + '.npy'
    str2 = 'net_trajectory_edge' + '_' + str(num1) + '_' + str(num2) + '.npy'
    np.save(str1, net_trajectory_node)
    np.save(str2, net_trajectory_edge)
    
