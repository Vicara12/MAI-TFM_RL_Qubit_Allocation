import json
import torch
import time
import pandas as pd
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
    times = {}
    for (name, circuit) in circuits.items():
        print(f'hqa optimizing {name}')
        t_ini = time.time()
        allocs, _ = allocator.optimize(circuit, hardware=hardware)
        times[name] = time.time() - t_ini
        check_sanity(allocs, circuit, hardware)
        costs[name] = sol_cost(allocations=allocs, core_con=hardware.core_connectivity)
    return costs, times


def get_results_russo(n_qubits: int, circuits: dict[str, Circuit], hardware: Hardware):
    with open(f'data/russo_alloc_results_{n_qubits}.json', 'r') as f:
        results = json.load(f)
    costs = {}
    for (name, allocs_list) in results['allocations'].items():
        allocs = torch.tensor(allocs_list, dtype=torch.int)
        check_sanity(allocs, circuits[name], hardware)
        costs[name] = sol_cost(allocations=allocs, core_con=hardware.core_connectivity)
    return costs, results['times']


def compare_sota(n_qubits: int):
    circuits = load_circuits(data_path=f'data/all_{n_qubits}.json', n_qubits=100)
    hardware = Hardware(
        core_capacities=torch.tensor([10]*10, dtype=torch.int),
        core_connectivity=(torch.ones([10,10]) - torch.eye(10)),
    )
    results = {
        'russo': get_results_russo(n_qubits, circuits=circuits, hardware=hardware),
        'hqa': get_results_hqa(circuits=circuits, hardware=hardware),
    }
    costs = {name: data[0] for name, data in results.items()}
    times = {name: data[1] for name, data in results.items()}

    costs = pd.DataFrame.from_dict(costs, orient="index").T
    times = pd.DataFrame.from_dict(times, orient="index").T

    costs.to_csv(f"data/sota_cost_{n_qubits}.csv", index=True)
    times.to_csv(f"data/sota_time_{n_qubits}.csv", index=True)


if __name__ == "__main__":
    compare_sota(50)
    compare_sota(100)