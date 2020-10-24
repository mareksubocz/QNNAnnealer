from annealer import get_qnn_bqm
import tabu
from pprint import pprint

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwavebinarycsp.exceptions import ImpossibleBQM

def start_annealing(layers: list, lagrange_propagation=1, qpu=False, stitch_kwargs=None):
    if qpu:
        sampler = EmbeddingComposite(
            DWaveSampler(solver={'topology__type': 'pegasus', 'qpu': True}))
    else:
        sampler = tabu.TabuSampler()

    bqm = get_qnn_bqm(layers, lagrange_propagation, stitch_kwargs)

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
    layers = [2,20,2]
    # qpu: use real qpu instead of local simulator
    start_annealing(layers, lagrange_propagation=1, qpu=False)

