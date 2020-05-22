import json
import pygsp
import numpy as np
import scipy as sp
import networkx as nx
from networkx.readwrite import json_graph
from sklearn import preprocessing

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

def label_to_vector(x):
    new_label = []
    for i in range(x.shape[0]):
        new_label.append(np.where(x[i]==1))
    return np.array(new_label)

def feature_broadcast(feats, G):
    new_feats = np.zeros(feats.shape)
    for i in range(feats.shape[0]):
        neighbors = list(G.neighbors(i))
        if len(neighbors) == 0:
            new_feats[i] = feats[i]
        else:
            surrounding = np.zeros(feats.shape[1])
            for j in neighbors:
                surrounding += feats[j]
            new_feats[i] = 0.5*feats[i]+0.5*surrounding/len(neighbors)

    return new_feats

def label_broadcast(G, labels, rate):
    remove_edges=[]
    for src, dst in G.edges():
        if labels[src] != labels[dst]:
            if np.random.random() <= rate:
                remove_edges.append((src, dst))
    for src, dst in remove_edges:
        G.remove_edge(src, dst)
    print('remove {} edges'.format(len(remove_edges)))
    return G

def remove_unlabeled(G):
    nids = set()
    for nid in G.nodes():
        if G.node[nid]['test'] and G.node[nid]['val']:
            nids.add(nid)
    for nid in nids:
        G.remove_node(nid)
    return G

def compute_feature_smoothness(path, times=0):
    G_org = json_graph.node_link_graph(json.load(open(path+'-G.json')))
    # G_org = remove_unlabeled(G_org)
    if nx.is_directed(G_org):
        G_org = G_org.to_undirected()
    edge_num = G_org.number_of_edges()
    G = pygsp.graphs.Graph(nx.adjacency_matrix(G_org))
    feats = np.load(path+'-feats.npy')
    # smooth
    for i in range(times):
        feats = feature_broadcast(feats, G_org)
    np.save(path+'-feats_'+str(times)+'.npy', feats)

    min_max_scaler = preprocessing.MinMaxScaler()
    feats = min_max_scaler.fit_transform(feats)
    smoothness = np.zeros(feats.shape[1])
    for src, dst in G_org.edges():
        smoothness += (feats[src]-feats[dst])*(feats[src]-feats[dst])
    smoothness = np.linalg.norm(smoothness,ord=1)
    print('The smoothness is: ', 2*smoothness/edge_num/feats.shape[1])

def compute_label_smoothness(path, rate=0.):
    G_org = json_graph.node_link_graph(json.load(open(path+'-G.json')))
    # G_org = remove_unlabeled(G_org)
    if nx.is_directed(G_org):
        G_org = G_org.to_undirected()
    class_map = json.load(open(path+'-class_map.json'))
    for k, v in class_map.items():
        if type(v) != list:
            class_map = convert_list(class_map)
        break
    labels = convert_ndarray(class_map)
    labels = np.squeeze(label_to_vector(labels))

    # smooth
    G_org = label_broadcast(G_org, labels, rate)
    with open(path+'-G_'+str(rate)+'.json', 'w') as f:
        f.write(json.dumps(json_graph.node_link_data(G_org)))

    edge_num = G_org.number_of_edges()
    G = pygsp.graphs.Graph(nx.adjacency_matrix(G_org))
    smoothness = 0
    for src, dst in G_org.edges():
        if labels[src] != labels[dst]:
            smoothness += 1
    print('The smoothness is: ', 2*smoothness/edge_num)

if __name__ == '__main__':
    # compute_feature_smoothness('../as/as')
    # compute_label_smoothness('../as/as')
