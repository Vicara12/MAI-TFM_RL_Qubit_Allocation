import json
from utils.customtypes import Circuit, Hardware
from sampler.randomcircuit import RandomCircuit


def gen_random(n_qubits: int, n_slices: int, n_circuits: int):
    rs = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
    return {f'random{i}': rs.sample().slice_gates for i in range(n_circuits)}


def load_circuits(circuit_names: list[str], n_qubits: int):
    files = {name: f'circuits/{name}_{n_qubits}.qasm' for name in circuit_names}
    return {name: Circuit.from_qasm(file, n_qubits).slice_gates for (name, file) in files.items()}


def gen_circuit_json(n_qubits: int = 50):
    circuit_names = [
        'cuccaro_adder',
        'deutsch_jozsa',
        'drapper_adder',
        'graph_state',
        'qft',
        'qnn',
        'quantum_volume',
    ]

    randoms = gen_random(n_qubits=n_qubits, n_slices=50, n_circuits=64)
    standard = load_circuits(circuit_names=circuit_names, n_qubits=n_qubits)
    data = {
        'n_qubits': n_qubits,
        'circuits': randoms | standard,
    }
    with open(f'data/all_{n_qubits}.json', 'w') as f:
        json.dump(data, f, indent=2)