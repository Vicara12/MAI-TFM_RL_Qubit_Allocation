import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    RealAmplitudes, EfficientSU2, TwoLocal, ExcitationPreserving, 
    ZFeatureMap, ZZFeatureMap, PauliFeatureMap,
    QFT, DraperQFTAdder, CDKMRippleCarryAdder, VBERippleCarryAdder,
    GroverOperator, IntegerComparator,
    PhaseEstimation, LinearFunction
)

from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit


def get_real_circuit(num_qubits, circuit_number):
    switch_cases = [
        # --- 1. Variational Ansatzes ---
        lambda n: RealAmplitudes(n, entanglement='full', reps=2),     # ID 0
        lambda n: RealAmplitudes(n, entanglement='linear', reps=3),   # ID 1
        lambda n: RealAmplitudes(n, entanglement='circular', reps=2), # ID 2
        
        lambda n: EfficientSU2(n, su2_gates=['ry'], entanglement='linear', reps=2), # ID 3
        lambda n: EfficientSU2(n, su2_gates=['rx', 'rz'], entanglement='full', reps=1), # ID 4
        lambda n: EfficientSU2(n, su2_gates=['h', 'rz'], entanglement='pairwise', reps=2), # ID 5
        
        lambda n: ExcitationPreserving(n, mode='iswap', entanglement='full'),   # ID 6
        lambda n: ExcitationPreserving(n, mode='fsim', entanglement='linear'), # ID 7
        
        lambda n: TwoLocal(n, 'rx', 'cx', entanglement='linear', reps=2),            # ID 8
        lambda n: TwoLocal(n, ['ry', 'rz'], 'cz', entanglement='full', reps=2),      # ID 9
        lambda n: TwoLocal(n, 'ry', 'cry', entanglement='circular', reps=1),         # ID 10
        lambda n: TwoLocal(n, 'h', 'cx', entanglement='sca', reps=2),                # ID 11
        
        # --- 2. Feature Maps ---
        
        lambda n: ZZFeatureMap(n, reps=1, entanglement='linear'),   # ID 14
        lambda n: ZZFeatureMap(n, reps=2, entanglement='full'),     # ID 15
        lambda n: ZZFeatureMap(n, reps=2, entanglement='circular'), # ID 16
        
        lambda n: PauliFeatureMap(n, reps=1, paulis=['X', 'Y', 'ZZ']), # ID 17
        lambda n: PauliFeatureMap(n, reps=2, paulis=['Z', 'XX']),      # ID 18

        # --- 3. Arithmetic & Logic ---
        lambda n: QFT(n, do_swaps=True),                # ID 20
        lambda n: QFT(n, approximation_degree=2),       # ID 21
        
        # Adders (Use n_half helper inside wrapper)
        lambda n: DraperQFTAdder(n // 2, kind='fixed'),          # ID 22
        lambda n: DraperQFTAdder(n // 2, kind='half'),           # ID 23
        lambda n: CDKMRippleCarryAdder(n // 2, kind='full'),     # ID 24
        lambda n: VBERippleCarryAdder(n // 2, kind='half'),      # ID 25
        
        lambda n: IntegerComparator(num_state_qubits=n // 2, value=1), # ID 26
        
        # --- 4. Textbook Algorithms (Manual Constructions) ---
        
        # Bernstein-Vazirani (Variant 1)
        lambda n: _build_bv(n), # ID 27
        # Bernstein-Vazirani (Variant 2 - randomness handled by seed above)
        lambda n: _build_bv(n), # ID 28
        # Bernstein-Vazirani (Variant 3)
        lambda n: _build_bv(n), # ID 29

        # Deutsch-Jozsa (Balanced)
        lambda n: _build_dj_balanced(n), # ID 31

        # Grover Operator
        lambda n: _build_grover(n), # ID 32
        
        # Simon's Algorithm
        lambda n: _build_simon(n), # ID 33

        # --- 5. Advanced / Library ---
        lambda n: PhaseEstimation(num_evaluation_qubits=n-1, unitary=_get_t_gate_unitary()), # ID 34
        
        lambda n: _build_hidden_shift(n), # ID 35
        
        # Graph States
        lambda n: _build_graph_chain(n), # ID 36
        lambda n: _build_graph_star(n),  # ID 37
        
        # Entangled States
        lambda n: _build_ghz(n),         # ID 38
        lambda n: _build_w_state(n),     # ID 39 (Only valid if n>=3, handled in helper)
    ]

    return switch_cases[circuit_number%len(switch_cases)](num_qubits)


def _get_t_gate_unitary():
    u = QuantumCircuit(1)
    u.p(np.pi/4, 0)
    return u

def _build_bv(n):
    # Bernstein-Vazirani
    s_bv = "".join([str(np.random.randint(0, 2)) for _ in range(n)])
    qc = QuantumCircuit(n + 1)
    qc.h(range(n + 1))
    qc.z(n)
    for i, char in enumerate(reversed(s_bv)):
        if char == '1':
            qc.cx(i, n)
    qc.h(range(n))
    return qc

def _build_dj_constant(n):
    qc = QuantumCircuit(n + 1)
    qc.h(range(n + 1))
    qc.z(n)
    qc.h(range(n))
    return qc

def _build_dj_balanced(n):
    qc = QuantumCircuit(n + 1)
    qc.h(range(n + 1))
    qc.z(n)
    for i in range(n):
        qc.cx(i, n)
    qc.h(range(n))
    return qc

def _build_grover(n):
    oracle = QuantumCircuit(n)
    # Simple oracle that marks state |1...1>
    if n > 1:
        oracle.h(n-1)
        oracle.mcx(list(range(n-1)), n-1)
        oracle.h(n-1)
    return GroverOperator(oracle)

def _build_simon(n):
    qc = QuantumCircuit(n * 2)
    qc.h(range(n))
    # Oracle for s=11...1
    for i in range(n):
        qc.cx(i, n + i)
    for i in range(n):
        qc.cx(n-1, n + i)
    qc.h(range(n))
    return qc

def _build_hidden_shift(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(0, n, 2): 
        if i+1 < n: qc.cz(i, i+1)
    qc.h(range(n))
    return qc

def _build_graph_chain(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n - 1):
        qc.cz(i, i+1)
    return qc

def _build_graph_star(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(1, n):
        qc.cz(0, i)
    return qc

def _build_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i+1)
    return qc

def _build_w_state(n):
    if n < 3: return QuantumCircuit(n) # Fallback
    qc = QuantumCircuit(3) # Hardcoded for 3 in original, kept as is
    qc.ry(2 * np.arccos(1/np.sqrt(3)), 0)
    qc.ch(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.x(0)
    # If n > 3 this simply returns a 3 qubit circuit on 'n' wires is invalid 
    # but strictly adhering to previous logic. 
    # Ideally should extend to N, but preserving user original logic:
    if n > 3:
        # Pad with identity
        qc_large = QuantumCircuit(n)
        qc_large.compose(qc, range(3), inplace=True)
        return qc_large
    return qc


class RealCircuit(CircuitSampler):
    def __init__(self, num_lq: int, max_slices: int):
        super().__init__(num_lq)
        self.max_slices = max_slices            
    
    def sample(self) -> Circuit:
        n_slices = 0
        while n_slices == 0:
            circuit = get_real_circuit(num_qubits=self.num_lq, circuit_number=np.random.randint(0,40))
            circuit = Circuit.from_qiskit(circuit, self.num_lq, cap_qubits=True)
            n_slices = circuit.n_slices
        slices = circuit.slice_gates
        if n_slices > self.max_slices:
            init_slice = np.random.randint(0,n_slices - self.max_slices)
            slices = slices[init_slice:(init_slice+self.max_slices)]
        return Circuit(slice_gates=slices, n_qubits=circuit.n_qubits)

    def __str__(self):
        return f'RealCircuit(num_lq={self.num_lq})'