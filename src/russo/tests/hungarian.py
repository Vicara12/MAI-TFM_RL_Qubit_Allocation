from scipy.optimize import linear_sum_assignment
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost, check_sanity
import numpy as np
import torch


def initial_assignement(slices, num_qubits, num_cores, capacity) -> tuple:
  assignement = np.full(num_qubits, -1)
  capacities = np.full(num_cores, capacity)
  for q1, q2 in slices[0]:
    for c in range(num_cores):
      if capacities[c] >= 2:
        assignement[q1] = c
        assignement[q2] = c
        capacities[c] -= 2
        break
  for q in range(num_qubits):
    if assignement[q] == -1:
      for c in range(num_cores):
        if capacities[c] >= 1:
          assignement[q] = c
          capacities[c] -= 1
          break
  return assignement, capacities


def hungarian(cost_matrix):
  gate_ids, core_ids = linear_sum_assignment(cost_matrix)
  cost = cost_matrix[gate_ids, core_ids].sum()
  return gate_ids, core_ids, cost


def calculate_attraction(slices, unfeasible_gates, current, t, num_qubits, num_cores):
    num_gates = len(unfeasible_gates)
    # Equation 20
    attr_q_t = np.zeros((num_qubits, num_cores), dtype=np.float64)
    for q in range(num_qubits):
      for c in range(num_cores):
        for q_ in range(num_qubits):
          if current[q_] == c:
            for m in range(t+1, len(slices)):
              if ((slices[m][:,0] == q) & (slices[m][:,1] == q_)).any() or ((slices[m][:,0] == q_) & (slices[m][:,1] == q)).any():
              #if ((slices[m][0] == q) & (slices[m][1] == q_)).any():
              #if (q, q_) in slices[m] or (q_, q) in slices[m]:
                attr_q_t[q, c] += 2.0 ** (t - m)

    # Equation 21
    attr_g_t = np.empty((num_gates, num_cores), dtype=np.float64)
    for g, (q1, q2) in enumerate(unfeasible_gates):
      attr_g_t[g] = (attr_q_t[q1] + attr_q_t[q2]) / 2
    
    return attr_g_t


def hungarian_assignement(
  slices,
  num_qubits,
  num_cores,
  capacity,
  lookahead=False,
  distance_matrix=None,
  initial=None,
  verbose=False,
):
  
  # Initial Assignment
  current, cur_capacities = initial_assignement(slices, num_qubits, num_cores, capacity)
  if initial is not None:
    current, cur_capacities = initial.copy(), capacity - np.bincount(initial, minlength=num_cores)
  assignments = [current.copy()]
  
  unfeasible_gates = []
  slice_arr = [np.array(s) for s in slices]
  
  for t in range(1, len(slices)):
    if verbose:
      print(f"\033[2K\r - Slice {t+1}/{len(slices)}", end='')
    for q1, q2 in slices[t]:
      if current[q1] != current[q2]:
        unfeasible_gates.append((q1, q2))
        # remove unfeasible gates from assignment
        cur_capacities[current[q1]] += 1
        cur_capacities[current[q2]] += 1  
        
    if True:
      """
      From: https://arxiv.org/pdf/2309.12182.pdf
        Each core must contain an even number of qubits interacting
        in unfeasible two-qubit gates for this approach to work.
        Otherwise, when assigning operations into cores, there will be
        a pair of qubits left to assign and two cores with exactly one
        free space each, making it impossible to assign an operation
        to the core. An auxiliary two-qubit gate involving two 
        noninteracting qubits from the cores with an odd number of
        qubits involved in unfeasible operations is created to solve
        this, ensuring that all cores contain an even number of free
        spaces and that all two-qubit operations will be allocated.
      """
      aux_gate = []
      used_qubits = np.array(slices[t]).flatten().tolist()
      for c in range(num_cores):
        if cur_capacities[c] % 2 == 1:
          for q in range(num_qubits):
            if current[q] == c and q not in used_qubits:
                cur_capacities[c] += 1
                aux_gate.append(q)
                break
      assert len(aux_gate) % 2 == 0, f"Error: Number of auxiliary gates is not even but is {len(aux_gate)}"
      if len(aux_gate) > 0:
        unfeasible_gates.extend([(aux_gate[i], aux_gate[i+1]) for i in range(0, len(aux_gate), 2)])
        
    while (num_gates := len(unfeasible_gates)) > 0:
      if lookahead:
          attr_g_t  = calculate_attraction(slice_arr, np.array(unfeasible_gates), current, t, num_qubits, num_cores)
      else:
          attr_g_t = np.zeros((num_gates, num_cores), dtype=np.float64)
      
      # Equation 22
      cost_matrix = np.zeros((num_gates, num_cores), dtype=np.float64)
      if distance_matrix is None:
        for g, (q1, q2) in enumerate(unfeasible_gates):
          for c in range(num_cores):
            if cur_capacities[c] == 0:
              cost_matrix[g, c] = 1e4
            elif current[q1] == c or current[q2] == c:
              cost_matrix[g, c] = 1 - attr_g_t[g, c]
            else:
              cost_matrix[g, c] = 2 - attr_g_t[g, c]
      else:
        for g, (q1, q2) in enumerate(unfeasible_gates):
          for c in range(num_cores):
            if cur_capacities[c] == 0:
              cost_matrix[g, c] = 1e4
            else:
              c1, c2 = current[[q1,q2]]
              cost_matrix[g, c] = distance_matrix[c1, c] + distance_matrix[c2, c] - attr_g_t[g, c]
      
      assert sum(cur_capacities % 2) == 0
      
      assigned_gates, assigned_cores, cost = hungarian(cost_matrix)
      unfeasible_gates_new = unfeasible_gates.copy()
      for g, c in zip(assigned_gates, assigned_cores):
        if cost_matrix[g, c] < 1e3:
          q1, q2 = unfeasible_gates[g]
          current[q1] = c
          current[q2] = c
          cur_capacities[c] -= 2
          unfeasible_gates_new.remove((q1, q2))
        
      unfeasible_gates = unfeasible_gates_new
    
    assert sum(cur_capacities) == 0, f"Error: Sum of cur_capacities is not 0 but {sum(cur_capacities)}"
    assignments.append(current.copy())
  if verbose:
    print('\033[2K\r', end='')
  return assignments


class HQA:
  def __init__(self, lookahead: bool, verbose: bool):
    self.lookahead = lookahead
    self.verbose = verbose

  def optimize(self, circuit: Circuit, hardware: Hardware):
    allocations = hungarian_assignement(
      slices=circuit.slice_gates,
      num_qubits=circuit.n_qubits,
      num_cores=hardware.n_cores,
      capacity=hardware.core_capacities.numpy(),
      lookahead=self.lookahead,
      distance_matrix=hardware.core_connectivity.numpy(),
      verbose=self.verbose
    )
    allocations = torch.tensor([s.tolist() for s in allocations], dtype=torch.int)
    check_sanity(allocs=allocations, circuit=circuit, hardware=hardware)
    cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
    return allocations, cost