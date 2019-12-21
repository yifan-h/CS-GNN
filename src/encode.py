# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/benedekrozemberczki/GraphWaveMachine
# which is under an identical MIT license

import networkx as nx
import numpy as np
import pygsp
import random
import pandas as pd
from pydoc import locate
import argparse

class WaveletMachine:
    """
    The class is a blue print for the procedure described in "Learning Structural Node Embeddings Via Diffusion Wavelets".
    """
    def __init__(self, G, node_id, sample_number):
        """
        This method 
        :param G: Input networkx graph object.
        """
        self.approximation = 100
        self.step_size = 20
        self.heat_coefficient = 1000.0
        self.node_label_type = str
        self.index=G.nodes()
        self.G = pygsp.graphs.Graph(nx.adjacency_matrix(G))
        self.number_of_nodes = len(nx.nodes(G))
        self.node_id = node_id
        self.sample_number = sample_number
        self.steps = [x*self.step_size for x in range(self.sample_number)]

    def approximate_wavelet_calculator(self):
        """
        Given the Chebyshev polynomial, graph the approximate embedding is calculated. 
        """
        self.real_and_imaginary = []
        for node in range(0,self.number_of_nodes):
            impulse = np.zeros((self.number_of_nodes))
            if node in self.node_id:
                impulse[node] = 1
                wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.G, self.chebyshev, impulse)
                self.real_and_imaginary.append([np.mean(np.exp(wavelet_coefficients*1*step*1j)) for step in self.steps])
            else:
                self.real_and_imaginary.append([0. for step in self.steps])
        self.real_and_imaginary = np.array(self.real_and_imaginary)


    def approximate_structural_wavelet_embedding(self):
        """
        Estimating the largest eigenvalue, setting up the heat filter and the Cheybshev polynomial. Using the approximate wavelet calculator method.
        """
        self.G.estimate_lmax()
        self.heat_filter = pygsp.filters.Heat(self.G, tau=[self.heat_coefficient])
        self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m=self.approximation)
        self.approximate_wavelet_calculator()

    def transform_and_save_embedding(self):
        """
        Transforming the numpy array with real and imaginary values to a pandas dataframe and saving it as a csv.
        """
        self.approximate_structural_wavelet_embedding()
        self.real_and_imaginary = np.concatenate([self.real_and_imaginary.real, self.real_and_imaginary.imag], axis=1)
        columns_1 = ["reals_" + str(x) for x in range(self.sample_number)]
        columns_2 = ["imags_" + str(x) for x in range(self.sample_number)]
        columns = columns_1 + columns_2
        self.real_and_imaginary = pd.DataFrame(self.real_and_imaginary, columns = columns)
        self.real_and_imaginary.index = self.index
        self.real_and_imaginary.index = self.real_and_imaginary.index.astype(locate(self.node_label_type))
        self.real_and_imaginary = self.real_and_imaginary.sort_index()

        return self.real_and_imaginary.values

def get_coding_feats(G, node_id, sample_number):
    node_id = set(node_id)
    total_node_id = sorted(G)
    G = nx.relabel.convert_node_labels_to_integers(G, ordering='sorted')
    new_node_id = []
    for i in range(len(total_node_id)):
        if total_node_id[i] in node_id:
            new_node_id.append(i)
    machine = WaveletMachine(G, new_node_id, sample_number)
    return machine.transform_and_save_embedding()
