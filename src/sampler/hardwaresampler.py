import torch
import random
from utils.customtypes import Hardware

class HardwareSampler:
  def __init__(self, max_nqubits: int, range_ncores: tuple[int,int]):
    self.max_nqubits = max_nqubits
    self.range_ncores = range_ncores
  
  def sample(self) -> Hardware:
    n_cores = random.randint(self.range_ncores[0], self.range_ncores[1])
    n_qubits = random.randint(2*n_cores, self.max_nqubits)
    while True:
      # We want all cores to have an even number of qubits to avoid impossible allocs
      core_caps = [2*random.randint(1,n_qubits//n_cores) for _ in range(n_cores)]
      if sum(core_caps) <= self.max_nqubits:
        n_qubits = sum(core_caps)
        break
    core_con = torch.ones(size=(n_cores,n_cores)) - torch.eye(n_cores)
    return Hardware(core_capacities=torch.tensor(core_caps, dtype=torch.int),
                    core_connectivity=core_con
    )