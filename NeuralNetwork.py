from random import random


class NeuralNetwork:
    def __init__(self, data_instance, edited_data, compressed_data, centroids_cluster, medoids_cluster):
        self.data_instance = data_instance
        self.edited_data = edited_data
        self.compressed_data = compressed_data
        self.centroids_cluster = centroids_cluster
        self.medoids_cluster = medoids_cluster

    def make_layers(self, no_of_layers, no_of_nodes):
        """
        :param no_of_layers: sets up the number of hidden layers for the network
        :param no_of_nodes: sets up the number of nodes in the hidden layer
        :return:
        """
        layers = []
        for index, row in self.data_instance:
            layers.append(row.drop(columns=self.data_instance.label_col))
            for i in range(no_of_layers + 1):
                layers.append(Layer(no_of_nodes))
                layers[i+1].make_nodes()
                for j in range(len(layers[i].nodes)):
                    for f in range(len(layers[i+1].nodes)):
                        layers[i].nodes[j].outgoing_weights.append(Weight(layers[i].nodes[j], layers[i+1].nodes[f]))
                        layers[i + 1].nodes[f].incoming_weights.append(Weight(layers[i].nodes[j], layers[i+1].nodes[f]))
        return layers

    def set_output(self):
        if not self.data_instance.regression:
            




class Layer:
    def __init__(self, no_of_nodes):
        self.no_of_nodes = no_of_nodes
        self.nodes = []

    def make_nodes(self):
        for nodes in range(self.no_of_nodes):
            self.nodes.append(Neuron(random(-.01, .01)))


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.is_sigmoidal = None
        self.is_linear = None
        self.incoming_weights = []
        self.outgoing_weights = []


class Weight:
    def __init__(self, L_neuron, R_neuron):
        self.L_neuron = L_neuron
        self.neuron2 = R_neuron
        self.weight = random(-.01, .01)

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight
