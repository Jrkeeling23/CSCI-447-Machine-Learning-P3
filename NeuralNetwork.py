import random
import math


class NeuralNetwork:
    def __init__(self, data_instance): #, edited_data, compressed_data, centroids_cluster, medoids_cluster):
        self.data_instance = data_instance
        # self.edited_data = edited_data
        # self.compressed_data = compressed_data
        # self.centroids_cluster = centroids_cluster
        # self.medoids_cluster = medoids_cluster

    def make_layers(self, no_of_layers, no_of_nodes):
        """
        :param no_of_layers: sets up the number of hidden layers for the network
        :param no_of_nodes: sets up the number of nodes in the hidden layer
        :return:
        """
        # row = self.data_instance.df.shape[0]-1
        first_layer_size = self.data_instance.df.shape[1]-1
        layers = []
        layers.append(Layer(first_layer_size))
        layers[0].make_nodes()
        for i in range(no_of_layers):
            layers.append(Layer(no_of_nodes))
            layers[i+1].make_nodes()
            if i+1 == no_of_layers:
                outputs = self.set_output()
                layers.append(Layer(len(outputs)))
                layers[i+2].make_nodes()
            #     print("sup")
            # print(i)
            # print(layers[1].no_of_nodes)
            for j in range(layers[i].no_of_nodes):
                for f in range(len(layers[i+1].nodes)):
                    layers[i].nodes[j].outgoing_weights.append(Weight(layers[i].nodes[j], layers[i+1].nodes[f]))
                    layers[i + 1].nodes[f].incoming_weights = layers[i].nodes[j].outgoing_weights
        for j in range(len(layers[-2].nodes)):
            for f in range(len(layers[-1].nodes)):
                layers[-2].nodes[j].outgoing_weights.append(Weight(layers[-2].nodes[j], layers[-1].nodes[f]))
                layers[-1].nodes[f].incoming_weights = layers[-2].nodes[j].outgoing_weights
        return layers, outputs



        # point_nets = []
        # for index, row in self.data_instance.train_df.iterrows():
        #     layers = []
        #     layers.append(Layer(len(row.drop(columns=self.data_instance.label_col))))
        #     layers[0].make_input_layer(row.drop(columns=self.data_instance.label_col))
        #     for i in range(no_of_layers):
        #         layers.append(Layer(no_of_nodes))
        #         layers[i+1].make_nodes()
        #         if i+1 == no_of_layers:
        #             layers.append(Layer(len(self.set_output())))
        #             layers[i+2].make_nodes()
        #         #     print("sup")
        #         # print(i)
        #         # print(layers[1].no_of_nodes)
        #         for j in range(layers[i].no_of_nodes):
        #             for f in range(len(layers[i+1].nodes)):
        #                 layers[i].nodes[j].outgoing_weights.append(Weight(layers[i].nodes[j], layers[i+1].nodes[f]))
        #                 layers[i + 1].nodes[f].incoming_weights = layers[i].nodes[j].outgoing_weights
        #     for j in range(len(layers[-2].nodes)):
        #         for f in range(len(layers[-1].nodes)):
        #             layers[-2].nodes[j].outgoing_weights.append(Weight(layers[-2].nodes[j], layers[-1].nodes[f]))
        #             layers[-1].nodes[f].incoming_weights = layers[-2].nodes[j].outgoing_weights
        #     point_nets.append(layers)

    def sigmoid(self, layers, input):
        # i = 0
        layers[0].make_input_layer(input)
        for layer in layers[1:]:
            for node in layer.nodes:
                sigmoid_total = 0
                for weight in node.incoming_weights:
                    # print(len(layers))
                    # print(len(layers[1:]))
                    # print(weight.get_weight())
                    sigmoid_total += weight.get_weight() \
                                     * weight.L_neuron.value
                sigmoid_total += node.bias
                node.value = 1/(1 + math.exp(-sigmoid_total))
        output = []
        for node in layers[-1].nodes:
            output.append(node.value)
        return output

    def prediction(self, outputs, output_values):
        guess = 0
        for i in range(len(outputs)):
            if output_values[i] > guess:
                guess = i
        return outputs[guess]

    def cost(self, output_values, outputs, expected):
        high_value = 0
        for i in range(len(outputs)):
            if outputs[i] == expected:
                high_value = i
        compare = []
        for j in range(len(output_values)):
            if j != high_value:
                compare.append(float(0))
            else:
                compare.append(float(1))
        cost = 0
        for f in range(len(output_values)):
            cost += (output_values[f]-compare[f]) ** 2
        return cost



    def set_output(self):
        output = []
        label = self.data_instance.label_col
        if not self.data_instance.regression:
            for index, row in self.data_instance.train_df.iterrows():
                if row[label] not in output:
                    output.append(row[label])
        return sorted(output)

    def back_prop(self):  # oh boy here we go wish me luck

        pass


class Layer:
    def __init__(self, no_of_nodes):
        self.no_of_nodes = no_of_nodes
        self.nodes = []

    def make_nodes(self):
        for nodes in range(self.no_of_nodes):
            self.nodes.append(Neuron(float(random.randint(-1, 1))/100))

    def make_input_layer(self, inputs):
        i = 0
        # print(len(self.nodes))
        for input in inputs:
            # print(input)
            self.nodes[i].value=input
            i+=1


class Neuron:
    def __init__(self, bias, value=None):
        self.bias = bias
        self.is_sigmoidal = None
        self.is_linear = None
        self.incoming_weights = []
        self.outgoing_weights = []
        self.value = value


class Weight:
    def __init__(self, L_neuron, R_neuron):
        self.L_neuron = L_neuron
        self.R_neuron = R_neuron
        self.weight = float(random.randint(-1, 1))/100

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight
