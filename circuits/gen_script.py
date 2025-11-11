import networkx as nx
from qiskit.circuit.random import random_circuit
from qiskit import qasm2, transpile

from qiskit.circuit.library import DraperQFTAdder, CDKMRippleCarryAdder, QuantumVolume, QFT, MCMTVChain, XGate, GraphState


seed  = 42

for num_qubits in [50,100]:
    c_random = random_circuit(num_qubits, 30, measure=False, seed=seed)
    c_qft = QFT(num_qubits, do_swaps=False, inverse=False).decompose()
    c_qv = QuantumVolume(num_qubits, seed=seed).decompose()

    p_edge = 0.5
    gs = nx.erdos_renyi_graph(num_qubits, p_edge, seed=seed, directed=False)
    gs_adj_matrix = nx.adjacency_matrix(gs).toarray()
    c_graph = GraphState(gs_adj_matrix).decompose()

    c_draper = DraperQFTAdder(num_qubits // 2, kind='fixed').decompose()
    c_cuccaro = CDKMRippleCarryAdder(num_qubits // 2 - 1, kind='fixed').decompose()

    def write_qasm(qc, filename: str) -> None:
        """Write the QuantumCircuit to a .qasm file (OpenQASM 2.0 string)."""
        qc = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=3, seed_transpiler=42)
        qasm_str = qasm2.dumps(qc)
        with open(filename, "w") as f:
            f.write(qasm_str)
        print(f"Wrote {filename} ({qc.num_qubits} qubits, {len(qc.data)} ops)")

    # write_qasm(c_qv, f'quantum_volume_{num_qubits}.qasm')
    write_qasm(c_cuccaro, f'cuccaro_adder_{num_qubits}.qasm')
    # write_qasm(c_graph, f'graph_state_{num_qubits}.qasm')
    write_qasm(c_draper, f'drapper_adder_{num_qubits}.qasm')
    # write_qasm(c_qft, f'qft_{num_qubits}.qasm')
