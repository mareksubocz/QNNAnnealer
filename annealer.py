import numpy as np
import dwavebinarycsp
from pyqubo import Binary, Array

def get_qnn_bqm(layers: list, input_data, output_data, lagrange_propagation=1, stitch_kwargs = None):
    if stitch_kwargs is None:
        stitch_kwargs = {}
    annealer = NeuralNetworkAnnealer(layers)
    return annealer.get_bqm(input_data, output_data, stitch_kwargs=stitch_kwargs, lagrange_propagation=lagrange_propagation)


class Network:
    def __init__(self, layers: list):
        self.layers = layers
        #FIXME: do poprawy inicjalizacja
        self.weights = [Array(
                    (np.random.rand(layers[i], layers[i+1]) * 2 - 1).tolist()
                ) for i in range(len(layers)-1)]
        #TODO: add bias?
        # self.biases = [np.random.rand(s) for s in layers]


class NeuralNetworkAnnealer:
    def __init__(self, layers):
        self.H = 0 # initialize Hamiltonian
        self.H_vars = set()
        self.network = Network(layers)


    def get_label(self, layer, num):
        return f"{layer}_{num}"


    def propagationConstraint(self, network: Network, input_data,
                              output_data, anneal_inputs=False,
                              lagrange = 1):
        # training data
        # TODO: pass training data by argument and process function in loop
        values = Array(input_data)
        expected = Array(output_data)

        # forward propagation
        for layer_num, layer in enumerate(network.layers[:-1]):

            # if you want to ensure input layer will not be annealed
            if layer_num == 0 and not anneal_inputs:
                values = values.dot(network.weights[layer_num])
                continue

            qubit_values = []
            nodes = {self.get_label(layer_num, i) for i in range(layer)}

            # TODO: delete zeros from matrices to speed up computation
            for node in nodes:
                # making sure the variable is in final equation
                if node not in self.H_vars:
                    x_var = Binary(node)
                    self.H_vars.add(x_var)
                else:
                    x_var = H_vars[node]

                qubit_values.append(x_var)
            print(self.H_vars)

            values *= Array(qubit_values)
            values = values.dot(network.weights[layer_num])

        # values is output array at this point
        self.H += sum((values - expected)*(values - expected))
        # * lagrange_propagation


    def get_bqm(self, input_trainig_data, output_training_data, stitch_kwargs = None, lagrange_propagation=1):
        if len(input_trainig_data) != len(output_training_data):
            print('Input and output training data length does not match')
        for tinput, toutput in zip(input_trainig_data, output_training_data):
            self.propagationConstraint(self.network, tinput, toutput,
                                       lagrange=lagrange_propagation)
        self.model = self.H.compile()
        bqm = self.model.to_dimod_bqm()
        return bqm
