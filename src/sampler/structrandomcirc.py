import torch
import random
from typing import Union
from utils.customtypes import Circuit
from sampler.circuitsampler import CircuitSampler



class StructuredRandomCircuit(CircuitSampler):

  def __init__(self, num_lq: int, slice_range: tuple[int,int], pattern_size: int, reflow: Union[bool,float] = False):
    super().__init__(num_lq)
    self.slice_range = slice_range
    self.pattern_size = pattern_size
    self.reflow = reflow


  def _new_qubit(self) -> int:
    # if not hasattr(self, 'lastq'):
    #   self.lastq = -1
    # self.lastq = (self.lastq + 1)%self.num_lq
    # return self.lastq
    return random.randint(0, self.num_lq-1)
  

  def _get_qubit(self, dist: torch.Tensor, prob_new: float, last_n: list[int], forbidden: list[int] = []) -> int:
    while True:
      # Get new qubit not in the sequence
      if torch.rand(1) < prob_new or dist.sum() == 0 or len(last_n) == 0:
        q = self._new_qubit()
        if q in last_n or q in forbidden:
          continue
        break
      # Repeat pattern
      else:
        idx = torch.distributions.Categorical(probs=dist/dist.sum()).sample().item()
        q = last_n[idx]
        if last_n[idx] in forbidden:
          continue
        break
    return q


  def _sample_gate(self):
    repeated = True
    while repeated:
      a = self._get_qubit(self.dist_a[-len(self.last_n_a):], self.prob_new_q_a, self.last_n_a)
      b = self._get_qubit(self.dist_b[-len(self.last_n_b):], self.prob_new_q_b, self.last_n_b, [a])
      repeated = self.last_gate is not None and self.last_gate in [(a,b), (b,a)]
    self.last_gate = (a,b)
    self.last_n_a.append(a)
    self.last_n_a = self.last_n_a[-self.pattern_size:]
    self.last_n_b.append(b)
    self.last_n_b = self.last_n_b[-self.pattern_size:]
    return (a,b)


  def _gen_prob_vec(self, smoothing: float, zero_last: bool):
    p = torch.randn(self.pattern_size).abs() + torch.linspace(self.pattern_size*smoothing,0,self.pattern_size)
    if zero_last:
      p[-1] = 0
    return p/p.sum()


  def sample(self) -> Circuit:
    self.lastq = -1
    self.last_n_a = []
    self.last_n_b = []
    self.prob_new_q_a = torch.rand(1).abs().item()*0.5 + 0.1
    self.prob_new_q_b = torch.rand(1).abs().item()*0.5 + 0.1
    smth = 0.5
    self.dist_a = self._gen_prob_vec(smoothing=smth, zero_last=False)
    self.dist_b = self._gen_prob_vec(smoothing=smth, zero_last=True)
    self.last_gate = None

    int_num_slices = random.randint(a=self.slice_range[0], b=self.slice_range[1])
    circuit_slice_gates = []
    a,b = self._sample_gate()
    for t in range(int_num_slices):
      used_qubits = set()
      slice_gates = []
      while not (a in used_qubits or b in used_qubits):
        slice_gates.append((a,b))
        used_qubits.add(a)
        used_qubits.add(b)
        a,b = self._sample_gate()
      circuit_slice_gates.append(tuple(slice_gates))
    if (self.reflow if isinstance(self.reflow, bool) else (random.randint(0,100) < 100*self.reflow)):
      gate_list = sum(circuit_slice_gates, tuple())
      return Circuit.from_gate_list(gates=gate_list, n_qubits=self.num_lq)
    return Circuit(slice_gates=tuple(circuit_slice_gates), n_qubits=self.num_lq)
  

  def __str__(self):
    return (f"StructuredRandomCircuit("
                f"num_lq={self.num_lq}, "
                f"slice_range={self.slice_range}, "
                f"last_n={self.pattern_size})")