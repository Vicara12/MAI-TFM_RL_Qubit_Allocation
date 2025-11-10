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

  for name in [
    # 'qft', # Gates ok, number of slices different but small
    # "quantum_volume", # Does not fit, unknown but small
    # 'graph_state', # Exact
    'drapper_adder', # VERY different -------------------
    'cuccaro_adder', # Different ------------------
    'qnn', # Massive difference ----------------------
    # 'deutsch_jozsa' # Exact
    ]:
    for nqubits in [50,100]:
      circuit = Circuit.from_qasm(f'circuits/{name}_{nqubits}.qasm')
      print(f"{name} {nqubits}: {circuit.n_slices} slices and {circuit.n_gates} gates")