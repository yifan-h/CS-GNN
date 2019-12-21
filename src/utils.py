import numpy as np
import time
import json
import collections
import os
import concurrent.futures as futures
import sklearn
import sklearn.cluster as sklc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph
from minibatch import NodeMinibatchIterator
from encode import get_coding_feats

seed = 123
np.random.seed(seed)

def process_graph(G):
    # Remove all nodes that do not have val/test annotations
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    if broken_count > 0:
        print("Removed {:d} nodes that lacked proper annotations.".format(broken_count))
    return G

def loadG(x, d):
    return json_graph.node_link_graph(json.load(open(x+'-G.json')), d)

def loadjson(x):
    return json.load(open(x))

def convert_dict(x, conv, lconv=int):
    return {conv(k):lconv(v) for k, v in x.items()}

def convert_ndarray(x):
    y = list(range(len(x)))
    for k, v in x.items():
        y[int(k)] = v
    return np.array(y)

def convert_list(x):
    c = []
    for k, v in x.items():
        if v not in c:
            c.append(v)
    new_x = {}
    for k, v in x.items():
        v_new = [0 for i in range(len(c))]
        v_new[c.index (v)] = 1
        new_x[k] = v_new
    return new_x

def check_rm(neighbors_set, unlabeled_nodes):
    for node in neighbors_set:
        if node not in unlabeled_nodes:
            return False
    return True

def rm_useless(G, feats, class_map, unlabeled_nodes, num_layers):
    # find useless nodes
    print('start to check and remove {} unlabeled nodes'.format(len(unlabeled_nodes)))
    unlabeled_nodes = set(unlabeled_nodes)
    rm_nodes = []
    for n_id in tqdm(unlabeled_nodes):
        neighbors_set = set()
        neighbors_set.add(n_id)
        for _ in range(num_layers):
            for node in neighbors_set:
                if nx.is_directed(G):
                    neighbors_set = neighbors_set | set(G.neighbors(node)) | set(G.predecessors(node))
                else:
                    neighbors_set = neighbors_set | set(G.neighbors(node))
        if check_rm(neighbors_set, unlabeled_nodes):
            rm_nodes.append(n_id)
    # rm nodes
    if len(rm_nodes):
        for node in rm_nodes:
            G.remove_node(node)
        G_new = nx.relabel.convert_node_labels_to_integers(G, ordering='sorted')
        feats = np.delete(feats, rm_nodes, 0)
        class_map = np.delete(class_map, rm_nodes, 0)
        print('remove {} '.format(len(rm_nodes)), 'useless unlabeled nodes')
    return G_new, feats, class_map



def load_data(prefix, num_layers=1, batch_size=1, concat=True, sample_number=50, directed=False):
    with futures.ProcessPoolExecutor(max_workers=5) as executor:
        # 1. read data
        start_time = time.time()
        futs = [executor.submit(loadG, prefix, directed),
                executor.submit(loadjson, prefix+'-class_map.json'),]
        if os.path.exists(prefix + '-feats.npy'):
            feats = np.load(prefix + '-feats.npy')
        else:
            feats = None

        # 2. process preparation
        class_map = futs[1].result()
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)
        G = futs[0].result()

        # 3. process data
        start_time = time.time()
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n
        fut = executor.submit(process_graph, G)
        class_map = convert_dict(class_map, conversion, lab_conversion)
        # if single label
        for k, v in class_map.items():
            if type(v) != list:
                class_map = convert_list(class_map)
            break
        G = fut.result()

        # 4. division
        start_time = time.time()
        train_nodes = [n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]
        val_nodes = [n for n in G.nodes() if not G.node[n]['test'] and G.node[n]['val']]
        test_nodes = [n for n in G.nodes() if G.node[n]['test'] and not G.node[n]['val']]
        unlabeled_nodes = [n for n in G.nodes() if G.node[n]['test'] and G.node[n]['val']]
        class_map = convert_ndarray(class_map)
        # remove useless nodes
        if len(unlabeled_nodes) > 0:
            G, feats, class_map = rm_useless(G, feats, class_map, unlabeled_nodes, num_layers)
            train_nodes = [n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]
            val_nodes = [n for n in G.nodes() if not G.node[n]['test'] and G.node[n]['val']]
            test_nodes = [n for n in G.nodes() if G.node[n]['test'] and not G.node[n]['val']]
            unlabeled_nodes = [n for n in G.nodes() if G.node[n]['test'] and G.node[n]['val']]    
        # double check
        if len(class_map) != len(train_nodes) + len(val_nodes) + len(test_nodes) + len(unlabeled_nodes):
            raise Exception('Error: repeat node id!')
        if max([n for n in G.nodes()]) != G.number_of_nodes()-1:
            raise Exception('Error: node id out of range!')

        # 5. encode topology features
        start_time = time.time()
        if os.path.exists(prefix + '-feats_t.npy'):
            feats_t = np.load(prefix + '-feats_t.npy')
            generate_tf = False
        else:
            feats_t = None
            generate_tf = True
      
        # 6. post process
        train_ids = np.array([n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
        if not generate_tf:
            train_feats_t = feats_t[train_ids]
            scaler.fit(train_feats_t)
            feats_t = scaler.transform(feats_t)
        print("load data in", "{:.5f}".format(time.time() - start_time), "seconds")

        # 7. minibatch
        print('start minibatch for train, val, test ...')
        start_time = time.time()
        G.remove_edges_from(G.selfloop_edges())
        train_subgraphs, train_subfeats, train_subfeats_t, train_sublabels, train_submasks = NodeMinibatchIterator(G, num_layers, \
                    batch_size, train_nodes, feats, feats_t, class_map, concat, generate_tf, prefix, sample_number).get_subgraphs()
        val_subgraphs, val_subfeats, val_subfeats_t, val_sublabels, val_submasks = NodeMinibatchIterator(G, num_layers, \
                    batch_size, val_nodes, feats, feats_t, class_map, concat, generate_tf, prefix, sample_number).get_subgraphs()
        test_subgraphs, test_subfeats, test_subfeats_t, test_sublabels, test_submasks = NodeMinibatchIterator(G, num_layers, \
                    batch_size, test_nodes, feats, feats_t, class_map, concat, generate_tf, prefix, sample_number).get_subgraphs()
        print('Done with minibatch within {:.5f} seconds, start training...'.format(time.time()-start_time))

    return G, train_subgraphs, train_subfeats, train_subfeats_t, train_sublabels, train_submasks, val_subgraphs, val_subfeats, \
        val_subfeats_t, val_sublabels, val_submasks, test_subgraphs, test_subfeats, test_subfeats_t, test_sublabels, test_submasks
