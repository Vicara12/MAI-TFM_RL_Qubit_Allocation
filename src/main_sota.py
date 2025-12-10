import json
import torch
from concurrent.futures import ProcessPoolExecutor
import functools

from russo.tests.hungarian import HQA
from utils.customtypes import Hardware, Circuit
from utils.allocutils import check_sanity, sol_cost

# Load circuits
def load_circuits(data_path: str, n_qubits: int):
    with open(data_path, 'r') as f:
        data = json.load(f)

    # n_qubits = data['n_qubits']
    circuit_slices = data['circuits']
    circuits = {}
    for (name, slices) in circuit_slices.items():
        circuits[name] = Circuit(slice_gates=slices, n_qubits=n_qubits)
    return circuits


def get_results_hqa(circuits: dict[str, Circuit], hardware: Hardware):
    allocator = HQA(lookahead=True, verbose=False)
    costs = {}
    for (name, circuit) in circuits.items():
        print(f'hqa optimizing {name}')
        allocs, _ = allocator.optimize(circuit, hardware=hardware)
        check_sanity(allocs, circuit, hardware)
        costs[name] = sol_cost(allocations=allocs, core_con=hardware.core_connectivity)
    return costs


def get_results_russo(n_qubits: int, circuits: dict[str, Circuit], hardware: Hardware):
    with open(f'data/russo_alloc_results_{n_qubits}.json', 'r') as f:
        results = json.load(f)
    costs = {}
    for (name, allocs_list) in results.items():
        allocs = torch.tensor(allocs_list, dtype=torch.int)
        check_sanity(allocs, circuits[name], hardware)
        costs[name] = sol_cost(allocations=allocs, core_con=hardware.core_connectivity)
    return costs


if __name__ == "__main__":
    circuits = load_circuits(data_path='data/all_100.json', n_qubits=100)
    hardware = Hardware(
        core_capacities=torch.tensor([10]*10, dtype=torch.int),
        core_connectivity=(torch.ones([10,10]) - torch.eye(10)),
    )
    results = {
        'russo': get_results_russo(100, circuits=circuits, hardware=hardware),
        'hqa': get_results_hqa(circuits=circuits, hardware=hardware),
    }
    algs = list(results.keys())
    csv = f"circuit,{','.join(algs)}\n"
    for circ in circuits.keys():
        circ_res = [str(int(results[alg][circ])) for alg in algs]
        csv += f"{circ},{','.join(circ_res)}\n"
    print('\nResults:\n')
    print(csv)