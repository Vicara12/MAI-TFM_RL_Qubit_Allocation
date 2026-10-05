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


class MultiTopologyHardwareSampler:
  def __init__(self, max_nqubits: int, range_ncores: tuple[int,int]):
    self.max_nqubits = max_nqubits
    self.range_ncores = range_ncores
    self.option = 0

  def sample(self) -> Hardware:
    n_cores = random.randint(self.range_ncores[0], self.range_ncores[1])
    n_qubits = random.randint(2*n_cores, self.max_nqubits)
    while True:
      # We want all cores to have an even number of qubits to avoid impossible allocs
      core_caps = [2*random.randint(1,n_qubits//n_cores) for _ in range(n_cores)]
      if sum(core_caps) <= self.max_nqubits:
        n_qubits = sum(core_caps)
        break
    if   self.option == 0: core_con = MultiTopologyHardwareSampler._coreConAllToAll(n_cores)
    elif self.option == 1: core_con = MultiTopologyHardwareSampler._coreConLinear(n_cores)
    elif self.option == 2: core_con = MultiTopologyHardwareSampler._coreConStar(n_cores)
    elif self.option == 3: core_con = MultiTopologyHardwareSampler._coreConRing(n_cores)
    else:                  core_con = MultiTopologyHardwareSampler._coreConGrid(n_cores)
    self.option = (self.option + 1)%5
    return Hardware(core_capacities=torch.tensor(core_caps, dtype=torch.int),
                    core_connectivity=core_con
    )

  @staticmethod
  def _coreConAllToAll(n_cores: int) -> torch.Tensor:
    return torch.ones(size=(n_cores,n_cores)) - torch.eye(n_cores)

  @staticmethod
  def _coreConLinear(n_cores: int) -> torch.Tensor:
    cons = torch.ones(size=(n_cores,n_cores))*float('inf')
    for i in range(n_cores):
      cons[i,i] = 0
      if i != (n_cores - 1):
        cons[i,i+1] = cons[i+1,i] = 1
    return Hardware.connectivityToDistance(cons)

  @staticmethod
  def _coreConStar(n_cores: int) -> torch.Tensor:
    cons = torch.ones(size=(n_cores,n_cores))*float('inf')
    for i in range(n_cores):
      cons[i,i] = 0
      if i != 0:
        cons[i,0] = cons[0,i] = 1
    return Hardware.connectivityToDistance(cons)

  @staticmethod
  def _coreConRing(n_cores: int) -> torch.Tensor:
    cons = torch.ones(size=(n_cores,n_cores))*float('inf')
    for i in range(n_cores):
      cons[i,i] = 0
      if i != (n_cores - 1):
        cons[i,i+1] = cons[i+1,i] = 1
    if n_cores > 2:
      cons[0, -1] = cons[-1, 0] = 1
    return Hardware.connectivityToDistance(cons)

  @staticmethod
  def _coreConGrid(n_cores: int) -> torch.Tensor:
    cons = torch.ones(size=(n_cores,n_cores))*float('inf')
    size = int(torch.ceil(torch.sqrt(torch.tensor(n_cores))))
    for i in range(n_cores):
      cons[i,i] = 0
      row = i//size
      col = i%size
      if col != 0:
        cons[i, i-1] = cons[i-1, i] = 1
      if col != (size-1) and i != (n_cores - 1):
        cons[i, i+1] = cons[i+1, i] = 1
      if row != 0:
        cons[i, i-size] = cons[i-size, i] = 1
      if row != (size-1) and i+size < n_cores:
        cons[i, i+size] = cons[i+size, i] = 1
    return Hardware.connectivityToDistance(cons)