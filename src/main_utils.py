import torch
from utils.customtypes import Circuit
from utils.allocutils import swaps_from_alloc, check_sanity_swap
from utils.plotter import drawQubitAllocation



if __name__ == '__main__':
  # drawQubitAllocation(
  #   qubit_allocation = torch.tensor([[0,0,1,1],[0,1,1,0],[0,1,1,0]]),
  #   core_capacities = torch.tensor([2,2]),
  #   circuit_slice_gates= (((0,1),(2,3)), ((1,2), (0,3)), ((1,2),)),
  #   file_name = 'example.svg',
  # )

  russo = dict(
    # qft=(99,197,1225,4950), # Exact
    quantum_volume=(50,100,1226,4961),
    # graph_state=(83,166,596,2449), # Exact
    drapper_adder=(120,245,925,3725),
    cuccaro_adder=(290,590,336,686), # A bit over
    qnn=(195,395,2498,9998),
    # deutsch_jozsa=(49,99,49,99), # Exact
  )

  for name, (s50, s100, g50, g100) in russo.items():
    circuit50 = Circuit.from_qasm(f'circuits/{name}_50.qasm')
    circuit100 = Circuit.from_qasm(f'circuits/{name}_100.qasm')
    print(f"{name}\n"
          f"   50: {circuit50.n_slices}|{s50} slices and {circuit50.n_gates}|{g50} gates\n"
          f"   100: {circuit100.n_slices}|{s100} slices and {circuit100.n_gates}|{g100} gates\n"
    )