import torch
import numpy as np
from typing import Tuple, Any
from scipy.optimize import linear_sum_assignment
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost



class HQA:
  ''' Hungarian Qubit Assignment (Escofet et al., 2023, https://arxiv.org/abs/2309.12182).

  Slice by slice, starting from the assignment of the previous slice, the qubits of the two-qubit
  gates that are split among different cores (unfeasible gates) are unassigned, and the gates are
  then distributed among the cores with the Hungarian algorithm, one gate per core and iteration,
  until all of them have been placed. The cost of placing gate op=(q_A,q_B) in core c is (Eq. 5):

      C_t(op, c) = moves(op, c) - (attr_t(q_A, c) + attr_t(q_B, c))/2      (inf if c has < 2 free spaces)

  where attr_t(q, c) is the sum of the lookahead weights w_t(q, q') = sum_{m>t} I(m,q,q') 2^-(m-t)
  over the qubits q' that are currently in core c (Eqs. 1 and 3).

  Unfeasible gates only fit if every core has an even number of free spaces. Cores with an odd number
  are fixed as in the paper: an idle qubit (not in any gate of the slice) is taken out of each of them
  and these qubits are paired into auxiliary gates that are assigned like the rest. The idle qubit with
  the lowest attraction to its core is chosen.

  Non all-to-all topologies (suggested as an extension in the paper): moving a qubit from core a to b
  counts D[a,b]/min(D>0) moves, the number of hops, which reduces to Eq. 5 for all-to-all.

  The paper starts from an already valid assignment. Here the cores are filled in qubit order and the
  first slice is then made valid as any other slice (the first assignment is not charged by sol_cost).
  '''

  def optimize(
    self,
    circuit: Circuit,
    hardware: Hardware,
    cfg: Any = None,
    verbose: bool = False
  ) -> Tuple[torch.Tensor, float]:
    ''' Same interface as DirectAllocator.optimize. `cfg` is ignored, HQA has no parameters. '''
    if circuit.n_qubits != hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {hardware.n_qubits} != {circuit.n_qubits}"
      ))
    n_slices, n_qubits = circuit.n_slices, circuit.n_qubits
    caps = hardware.core_capacities.cpu().numpy()
    dist = hardware.core_connectivity.cpu().numpy().astype(float)
    if (dist > 0).any():
      dist = dist / dist[dist > 0].min()
    # embedding[t] = sum_{m>=t} adj[m] 2^-(m-t+1), so embedding[t+1] are the lookahead weights w_t
    embs = circuit.embedding.cpu().numpy()
    no_lookahead = np.zeros((n_qubits, n_qubits), dtype=embs.dtype)

    allocations = np.empty((n_slices, n_qubits), dtype=np.int64)
    assign = np.repeat(np.arange(hardware.n_cores), caps)
    for t, gates in enumerate(circuit.slice_gates):
      w = embs[t+1] if t+1 < n_slices else no_lookahead
      assign = self._assign_slice(assign, gates, w, dist, caps)
      allocations[t] = assign
      if verbose:
        print(f"\033[2K\r - Optimization step {t+1}/{n_slices}", end="")
    if verbose:
      print()

    allocations = torch.from_numpy(allocations).to(torch.int)
    cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
    return allocations, cost


  @staticmethod
  def _attraction(assign: np.ndarray, w: np.ndarray, n_cores: int) -> np.ndarray:
    ''' attr[q,c] = sum of w[q,q'] over the qubits q' currently assigned to core c (Eq. 3). '''
    placed = np.flatnonzero(assign >= 0)
    in_core = np.zeros((len(assign), n_cores), dtype=w.dtype)
    in_core[placed, assign[placed]] = 1
    return w @ in_core


  def _assign_slice(
    self,
    prev: np.ndarray,
    gates,
    w: np.ndarray,
    dist: np.ndarray,
    caps: np.ndarray,
  ) -> np.ndarray:
    n_cores = len(caps)
    assign = prev.copy()
    ops = [tuple(g) for g in gates if prev[g[0]] != prev[g[1]]]
    if not ops:
      return assign
    for op in ops:
      assign[list(op)] = -1
    free = caps - np.bincount(assign[assign >= 0], minlength=n_cores)

    # Make the free spaces of every core even with auxiliary gates of idle qubits
    odd = np.flatnonzero(free % 2)
    if len(odd):
      assert len(odd) % 2 == 0, f"Odd number of cores with odd free spaces: {odd.tolist()}"
      busy = set(q for g in gates for q in g)
      attr = self._attraction(assign, w, n_cores)
      aux = []
      for c in odd:
        idle = [q for q in np.flatnonzero(assign == c) if q not in busy]
        if not idle:
          raise Exception(f"Core {c} has an odd number of free spaces and no idle qubit to move out")
        q = min(idle, key=lambda q: attr[q, c])
        assign[q] = -1
        free[c] += 1
        aux.append(q)
      ops += list(zip(aux[::2], aux[1::2]))

    # Assign one gate per core and iteration until all are placed
    ops = np.array(ops)
    while len(ops):
      cores = np.flatnonzero(free >= 2)
      if not len(cores):
        raise Exception(f"No core with two free spaces left for {len(ops)} gates")
      attr = self._attraction(assign, w, n_cores)
      moves = dist[prev[ops[:, 0]]][:, cores] + dist[prev[ops[:, 1]]][:, cores]
      cost = moves - 0.5*(attr[ops[:, 0]][:, cores] + attr[ops[:, 1]][:, cores])
      rows, cols = linear_sum_assignment(cost)
      for r, c in zip(rows, cols):
        assign[ops[r]] = cores[c]
        free[cores[c]] -= 2
      ops = np.delete(ops, rows, axis=0)
    return assign
