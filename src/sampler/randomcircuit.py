import random
from typing import Union, Callable
from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit


class RandomCircuit(CircuitSampler):
  ''' Random Circuit Sampler.

  Args:
    - num_lq: number of logical qubits of the circuit.
    - num_slices: number of slices of the generated circuit. If set to int its value is fixed, if
        set to a callable object then the number of slices each time sample() is called is determined
        by the returned value (i.e. lambda: randint(1,10)).
  '''
  def __init__(self, num_lq: int, num_slices: Union[int, tuple[int,int]], reflow: Union[bool,float] = False):
    super().__init__(num_lq)
    self.num_slices = num_slices if not isinstance(num_slices, int) else [num_slices,num_slices]
    self.reflow = reflow
  

  def sample(self) -> Circuit:
    int_num_slices = random.randint(*self.num_slices)
    circuit_slice_gates = []
    a,b = random.sample(range(0,self.num_lq_),2)
    for t in range(int_num_slices):
      used_qubits = set()
      slice_gates = []
      while not (a in used_qubits or b in used_qubits):
        slice_gates.append((a,b))
        used_qubits.add(a)
        used_qubits.add(b)
        a,b = random.sample(range(0,self.num_lq_),2)
      circuit_slice_gates.append(tuple(slice_gates))
    if (self.reflow if isinstance(self.reflow, bool) else (random.randint(0,100) < 100*self.reflow)):
      gate_list = sum(circuit_slice_gates, tuple())
      return Circuit.from_gate_list(gates=gate_list, n_qubits=self.num_lq)
    return Circuit(slice_gates=tuple(circuit_slice_gates), n_qubits=self.num_lq)
  

  def __str__(self):
    if self.num_slices is Callable:
      ns = str(self.num_slices())
    else:
      ns = str(self.num_slices)
    return f"RandomCircuit(num_lq={self.num_lq}, num_slices={ns}, reflow={self.reflow})"



class HotRandomCircuit(CircuitSampler):
  '''
  Random circuit in which one random qubit interacts very intensely.
  '''
  def __init__(self, num_lq: int, num_slices: Union[int, Callable[[], int]]):
    super().__init__(num_lq)
    self.num_slices = num_slices
  
  def sample(self) -> Circuit:
    int_num_slices = self.num_slices() if callable(self.num_slices) else self.num_slices
    qubits = list(range(self.num_lq))
    random.shuffle(qubits)
    circuit_slice_gates = [((qubits[0],qubits[1+(n%(self.num_lq-1))]),) for n in range(int_num_slices)]
    return Circuit(slice_gates=tuple(circuit_slice_gates), n_qubits=self.num_lq)
  
  def __str__(self):
    if self.num_slices is Callable:
      ns = str(self.num_slices())
    else:
      ns = str(self.num_slices)
    return f"HotRandomCircuit(num_lq={self.num_lq}, num_slices={ns})"



class DenseRandomCircuit(CircuitSampler):
  ''' Random Circuits with many gates per slice.
  '''
  def __init__(self, num_lq: int, num_slices: Union[int, Callable[[], int]]):
    super().__init__(num_lq)
    self.num_slices = num_slices
  

  def sample(self) -> Circuit:
    int_num_slices = self.num_slices() if callable(self.num_slices) else self.num_slices
    gate_list = []
    a,b = random.sample(range(0,self.num_lq_),2)
    for t in range(int_num_slices):
      qubits = list(range(2*(self.num_lq//2)))
      random.shuffle(qubits)
      gates = list(zip(qubits[::2],qubits[1::2]))
      gate_list += gates
    # Reflow
    return Circuit.from_gate_list(gates=gate_list, n_qubits=self.num_lq)
  

  def __str__(self):
    if self.num_slices is Callable:
      ns = str(self.num_slices())
    else:
      ns = str(self.num_slices)
    return f"SparseRandomCircuit(num_lq={self.num_lq}, num_slices={ns})"