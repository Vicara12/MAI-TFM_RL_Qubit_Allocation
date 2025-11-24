import random
import torch
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
  def __init__(self, num_lq: int, num_slices: Union[int, Callable[[], int]]):
    super().__init__(num_lq)
    self.num_slices = num_slices
  

  def sample(self) -> Circuit:
    int_num_slices = self.num_slices() if callable(self.num_slices) else self.num_slices
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
    return Circuit(slice_gates=tuple(circuit_slice_gates), n_qubits=self.num_lq)
  

  def __str__(self):
    if self.num_slices is Callable:
      ns = str(self.num_slices())
    else:
      ns = str(self.num_slices)
    return f"RandomCircuit(num_lq={self.num_lq}, num_slices={ns})"