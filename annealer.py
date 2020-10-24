import numpy as np
import dwavebinarycsp
from pyqubo import Binary, Array

def get_qnn_bqm(layers: list, lagrange_propagation=1, stitch_kwargs = None):
    if stitch_kwargs is None:
        stitch_kwargs = {}
    annealer = NeuralNetworkAnnealer(layers)
    return annealer.get_bqm(stitch_kwargs=stitch_kwargs, lagrange_propagation=lagrange_propagation)


class Network:
    def __init__(self, layers: list):
        self.layers = layers
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


    def propagationConstraint(self, network: Network, anneal_inputs=False,
                              lagrange = 1):
        # training data
        # TODO: pass training data by argument and process function in loop
        values = Array([0,1])
        expected = Array([1,0])

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

            values *= Array(qubit_values)
            values = values.dot(network.weights[layer_num])

        # values is output array at this point
        self.H += sum((values - expected)*(values - expected))


    def get_bqm(self, stitch_kwargs = None, lagrange_propagation=1):
        self.propagationConstraint(self.network, lagrange=lagrange_propagation)
        self.model = self.H.compile()
        bqm = self.model.to_dimod_bqm()
        return bqm
