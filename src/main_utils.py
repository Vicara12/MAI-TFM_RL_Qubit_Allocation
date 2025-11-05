import torch
from utils.allocutils import swaps_from_alloc, check_sanity_swap
from utils.plotter import drawQubitAllocation

if __name__ == '__main__':
  drawQubitAllocation(
    qubit_allocation = torch.tensor([[0,0,1,1],[0,1,1,0],[0,1,1,0]]),
    core_capacities = torch.tensor([2,2]),
    circuit_slice_gates= (((0,1),(2,3)), ((1,2), (0,3)), ((1,2),)),
    file_name = 'example.svg',
  )