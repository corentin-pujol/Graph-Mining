import numpy as np  
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities

def mask_most_centrale_nodes(centrality, factor=0.001):
    """ Return a mask of the most central nodes
    """
    cursor = 1
    step = 0.001
    max_node = int(len(centrality)*factor) + 1

    selected_nodes = 0
    while np.sum(selected_nodes) <=  max_node:
        selected_nodes = centrality > cursor
        cursor -= step
        
    return selected_nodes

# return a mask of 20 biggest value of an array
def mask_biggest_nodes(centrality, nb_node):
    """ Return a mask of the most central nodes
    """
    mask = np.zeros(len(centrality), dtype=bool)
    for _ in range(nb_node):
        mask[np.argmax(centrality)] = True
        centrality[np.argmax(centrality)] = 0
    return mask


def LC_diffusion(g, seeds, threshold=None, random_seed=None):
    """ Linear Threshold diffusion
    """
    # init
    nodes = np.array(list(g.nodes()))
    activated_nodes = np.zeros(nodes.size)
    activated_nodes[seeds] = 1
    activated_nodes = activated_nodes.astype(bool)
    
    if threshold is None:
        if random_seed is not None: np.random.seed(random_seed)
        threshold_nodes = np.random.random(nodes.size)
    else:
        threshold_nodes = threshold*np.ones(nodes.size)

    history = [seeds]
   
    # diffusion
    last_activated_node_number = np.sum(activated_nodes) - 1
    while np.sum(activated_nodes) > last_activated_node_number:

        last_activated_node_number = np.sum(activated_nodes)

        for i in range(nodes.size):
            if activated_nodes[i] == False:
                value = np.sum(activated_nodes[list(g.neighbors(nodes[i]))]) / len(list(g.neighbors(nodes[i])))
                activated_nodes[i] = value > threshold_nodes[i]
        
        history.append(nodes[activated_nodes])
    
    return history

def IC_diffusion(gg, seeds, threshold, random_seed=None):
    """ Independent Cascade diffusion
    """
    # init
    g = gg.copy()
    nodes = np.array(list(g.nodes()))
    activated_nodes = np.zeros(nodes.size)
    activated_nodes[seeds] = 1
    activated_nodes = activated_nodes.astype(bool)

    activated_nodes = set(seeds)
    
    if random_seed is not None: np.random.seed(random_seed)

    history = [activated_nodes]

    while len(g.nodes()) > 0:

        new_activated_nodes = set()
        new_desactivated_nodes = set()
        
        for node in activated_nodes:
            for neighbor in g.neighbors(node):
                if neighbor not in activated_nodes and neighbor not in new_activated_nodes:
                    if np.random.random() < threshold:
                        new_activated_nodes.add(neighbor)
                        if neighbor in new_desactivated_nodes:
                            new_desactivated_nodes.remove(neighbor)
                    else:
                        new_desactivated_nodes.add(neighbor)

        if len(new_activated_nodes) == 0:
            break
    
        h_set = history[-1].union(new_activated_nodes)
        history.append(h_set)

        g.remove_nodes_from(activated_nodes.union(new_desactivated_nodes))
        activated_nodes = new_activated_nodes.copy()
    return history

def inhomogeneous_list_to_array(l):
    """ transform an inhmogeneous list of list of list into a 3D array
    """
    max_len = max([len(x) for x in l])
    max_len2 = max([max([len(x) for x in l[k]]) for k in range(len(l))])
    arr = np.full((len(l), max_len, max_len2), np.nan)
    for i, x in enumerate(l):
        for j, y in enumerate(x):
            arr[i, j, :len(y)] = y
    return arr


#transform an array of indices into a boolean mask
def indices_to_mask(indices, n):
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return mask

def indices_to_color(indices, n, c1='c', c2='r'):
    mask = np.full(n, c1)
    mask[indices] = c2
    return mask






def louvain_dict(g, resolution):

    l_comm = louvain_communities(g, resolution=resolution)

    print("Nb comm : ", len(l_comm))
    for comm in l_comm:
        print(len(comm), end=", ")
    print("\n")

    return l_comm

def comm_dict(g, l_comm):
    # COMM_DICT
    #for each community, we compute the centrality of each node with difffrent centrality measures
    comm_dict = {}
    i = 0
    for comm in l_comm:
        list_comm = list(comm)
        comm_dict[i] = {"len" : len(comm), "nodes" : list_comm, 
                "degree" : list(nx.degree_centrality(g.subgraph(list_comm)).values()), 
                "closeness" : list(nx.closeness_centrality(g.subgraph(list_comm)).values()), 
                "betweenness" : list(nx.betweenness_centrality(g.subgraph(list_comm)).values()), 
                "eigenvector" : list(nx.eigenvector_centrality(g.subgraph(list_comm)).values())}
        i += 1
        
    return comm_dict

def central_dict(comm_dict, nb_node):     

    # CENTRAL_DICT #select most central node for each community
    central_dict = {}

    centralitys = ['degree', 'closeness', 'betweenness', 'eigenvector']
    for centrality in centralitys:
        central_dict[centrality] = {}
        for j in range(len(comm_dict)):
            arr = np.array(comm_dict[j][centrality])
            #mask_central_node = mask_most_centrale_nodes(arr, factor=factor)
            #print(np.sum(mask_central_node), end=", ")
            mask_central_node = mask_biggest_nodes(arr, nb_node=nb_node)
            central_nodes = np.array(comm_dict[j]["nodes"])[mask_central_node]
            central_dict[centrality][j] = central_nodes
        central_dict[centrality] = np.concatenate(list(central_dict[centrality].values()))

    for centrality in centralitys:
        print(centrality, len(central_dict[centrality]))

    return central_dict

def one_diffusion(g, central_dict):

## ONE DIFFUSION diffusion_dict
    diffusion_dict = {}
    diffusion_dict['LC'] = {}
    diffusion_dict['IC'] = {}

    centralitys = ['degree', 'closeness', 'betweenness', 'eigenvector']
    for centrality in centralitys:
        diffusion_dict['LC'][centrality] = LC_diffusion(g, central_dict[centrality])
        diffusion_dict['IC'][centrality] = IC_diffusion(g, central_dict[centrality], threshold=0.1)
    
    return diffusion_dict



## MULTIPLE DIFFUSION/EPISODES
def multiple_diffusion(g, central_dict, n=200):

    LC_list = []
    IC_list = []

    centralitys = ['degree', 'closeness', 'betweenness', 'eigenvector']

    for _ in range(n):
        LC_list_tmp = []
        IC_list_tmp = []
        for i, centrality in enumerate(centralitys):
            lc = LC_diffusion(g, central_dict[centrality])
            ic = IC_diffusion(g, central_dict[centrality], threshold=0.2)

            LC_list_tmp.append([len(x) for x in lc])
            IC_list_tmp.append([len(x) for x in ic])
        
        LC_list.append(LC_list_tmp)
        IC_list.append(IC_list_tmp)

    LC_arr = inhomogeneous_list_to_array(LC_list)
    IC_arr = inhomogeneous_list_to_array(IC_list)

    return LC_arr, IC_arr




def centrality_graph(diffusion_dict, diffusion_model='LC'):
    centralitys = ['degree', 'closeness', 'betweenness', 'eigenvector']
    for centrality in centralitys:
        lc = diffusion_dict['LC'][centrality]
        plt.plot([len(x) for x in lc])
    plt.legend(centralitys)


