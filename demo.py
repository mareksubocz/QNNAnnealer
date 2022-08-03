from annealer import get_qnn_bqm
import tabu
from pprint import pprint

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler


def start_annealing(layers: list, input_data, output_data,
                    lagrange_propagation=1, qpu=False, stitch_kwargs=None):
    if qpu:
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        sampler = tabu.TabuSampler()

    bqm = get_qnn_bqm(layers, input_data, output_data,
                      lagrange_propagation, stitch_kwargs)

    # Check elements in the BQM
    # for q in bqm.linear:
    #     if bqm.linear[q] != -1:
    #         print(q)
    # for q in bqm.quadratic:
    #     if bqm.quadratic[q] != 2:
    #         print(q, bqm.quadratic[q])

    # Run BQM and get the solution and the energy
    sampleset = sampler.sample(bqm, num_reads=1000)
    solution1 = sampleset.first.sample
    energy1 = sampleset.first.energy

    print("Nodes chosen ('layer_n.o'): ")
    pprint(solution1)
    print("Solution energy: ", energy1)

    # Determine which nodes are involved
    # selected_nodes = [k for k, v in solution1.items() if v == 1 and not k.startswith('aux')]
    # from pprint import pprint
    # pprint(selected_nodes, ' Energy ', energy1)


if __name__ == "__main__":
    layers = [2, 10, 1]

    # XOR function
    input_data = [[1, 0], [1, 1], [0, 1], [0, 0]]
    output_data = [[1], [0], [1], [0]]
    # AND function
    # input_data = [[1, 0], [1, 1], [0, 1], [0, 0]]
    # output_data = [[0], [1], [0], [0]]
    # input_data = [[1,0]]
    # output_data = [[1]]
    # qpu: use real qpu instead of local simulator
    start_annealing(layers, input_data, output_data, lagrange_propagation=1,
                    qpu=False)
