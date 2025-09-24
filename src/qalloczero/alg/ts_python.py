from typing import List, Self, Tuple, Dict, Optional
from math import sqrt, log
from dataclasses import dataclass
import torch
from qalloczero.models.inferenceserver import InferenceServer
from qalloczero.alg.ts import TSEngine, TSConfig, TSTrainData



class TSPythonEngine(TSEngine):
  ''' Python backend for the tree search engine.
  '''

  @dataclass
  class Node:
    # Circuit sate attributes
    current_allocs: torch.Tensor = None
    prev_allocs: torch.Tensor = None # Allocations in the previous time slice
    core_caps: torch.Tensor = None
    # State attributes
    allocation_step: int = None
    terminal: bool = False
    current_slice: int = None
    # RL attributes
    policy: torch.Tensor = None
    value_sum: float = None
    visit_count: int = 0
    children: Dict[int, Tuple[Self, float]] = None

    @property
    def expanded(self) -> bool:
      return self.children is not None

    @property
    def value(self) -> float:
      return self.value_sum/self.visit_count if not self.terminal else 0



  def __init__(
    self,
    n_qubits: int,
    core_caps: torch.Tensor,
    core_conns: torch.Tensor,
    verbose: bool = False,
  ):
    self.n_qubits = n_qubits
    self.core_caps = core_caps
    self.core_conns = core_conns
    self.n_cores = core_conns.shape[0]
    self.verbose = verbose
    

  def load_model(self, name: str, model: torch.nn.Module):
    return InferenceServer.addModel(name, model)
  

  def has_model(self, name: str):
    return InferenceServer.hasModel(name)
  

  def rm_model(self, name: str):
    return InferenceServer.removeModel(name)
  

  def optimize(
    self,
    slice_adjm: torch.Tensor,
    circuit_embs: torch.Tensor,
    alloc_steps: torch.Tensor,
    cfg: TSConfig,
    ret_train_data: bool
  ) -> Tuple[torch.Tensor, int, float, Optional[TSTrainData]]:
    allocs, tdata = self._init_opt(
      slice_adjm, circuit_embs, alloc_steps, cfg, ret_train_data)
    n_expanded_nodes = 0

    for step in range(self.n_steps):
      slice_idx = self.alloc_steps[self.root.allocation_step][0].item()
      qubit0 = self.alloc_steps[self.root.allocation_step][1].item()
      qubit1 = self.alloc_steps[self.root.allocation_step][2].item()

      if self.verbose:
        print(f" - Optimization step {step+1}/{self.n_steps}")

      if ret_train_data:
        self._store_train_data(tdata, step, slice_idx, qubit0, qubit1)
      
      (action, action_cost, logits, n_sims) = self._iterate()

      n_expanded_nodes += n_sims
      allocs[slice_idx, qubit0] = action
      if qubit1 != -1:
        allocs[slice_idx, qubit1] = action
      
      if ret_train_data:
        tdata.logits[step] = logits
        tdata.value[step] = action_cost
    
    if ret_train_data:
      for step in range(self.n_steps-2,-1,-1):
        tdata.value[step] += tdata.value[step+1]
        # Normalize by dividing by remaining gates
        tdata.value[step+1] /= self.alloc_steps[step+1][3]
      tdata.value[0] /= self.alloc_steps[0][3]
    
    expl_r = self._exploration_ratio(n_expanded_nodes)
    self.root = None
    self.slice_adjm = None
    self.circuit_embs = None
    self.alloc_steps = None
    return allocs, n_expanded_nodes, expl_r, tdata


  def _exploration_ratio(self, n_exp_nodes: int) -> float:
    theoretical_n_exp_nodes = (
      self.n_steps * self.cfg.target_tree_size * (self.n_cores - 1)/self.n_cores
    )
    return n_exp_nodes/theoretical_n_exp_nodes

    
  def _store_train_data(
    self,
    tdata: TSTrainData,
    step: int,
    slice_idx: int,
    qubit0: int,
    qubit1: int
  ):
    tdata.qubits[step,0] = qubit0
    tdata.qubits[step,1] = qubit1
    tdata.slice_idx[step,0] = slice_idx
    tdata.prev_allocs[step] = self.root.prev_allocs
    tdata.curr_allocs[step] = self.root.current_allocs
    tdata.core_caps[step] = self.root.core_caps


  def _init_opt(
    self,
    slice_adjm: torch.Tensor,
    circuit_embs: torch.Tensor,
    alloc_steps: torch.Tensor,
    cfg: TSConfig,
    ret_train_data: bool
  ) -> Tuple[torch.Tensor, TSTrainData]:
    self.slice_adjm = slice_adjm
    self.circuit_embs = circuit_embs
    self.alloc_steps = alloc_steps
    self.cfg = cfg
    self.n_slices = slice_adjm.shape[0]
    self.n_steps = alloc_steps.shape[0]
    self.root = self.__buildRoot()
    allocs = torch.empty([self.n_slices, self.n_qubits], dtype=torch.int)
    if ret_train_data:
      tdata = TSTrainData(
        qubits=torch.empty([self.n_steps, 2], dtype=torch.int),
        prev_allocs=torch.empty([self.n_steps, self.n_qubits], dtype=torch.long),
        curr_allocs=torch.empty([self.n_steps, self.n_qubits], dtype=torch.long),
        core_caps=torch.empty([self.n_steps, self.n_cores], dtype=torch.long),
        slice_idx=torch.empty([self.n_steps, 1], dtype=torch.long),
        logits=torch.empty([self.n_steps, self.n_cores], dtype=torch.float),
        value=torch.empty([self.n_steps, 1], dtype=torch.float),
      )
      return allocs, tdata
    return allocs, None


  def _iterate(self) -> Tuple[int, float, torch.Tensor, int]:
    # Visit count is equal to the current size of the tree (excluding non-expanded nodes)
    num_sims = self.cfg.target_tree_size - self.root.visit_count
    for _ in range(num_sims):
      node = self.root
      search_path = [(node, -1)]
      while node.expanded and not node.terminal:
        node, action = self.__selectChild(current_node=node)
        search_path.append((node, action))
      if not node.terminal:
        self._expandNode(node)
      self.__backprop(search_path)
    action, logits = self.__selectAction(self.root, self.cfg.action_sel_temp)
    action_cost = self.root.children[action][1]
    self.root = self.root.children[action]
    return action, action_cost, logits, num_sims


  @staticmethod
  def __selectAction(node: Node, temp: float) -> Tuple[int, torch.Tensor]:
    visit_counts = list(
      node.children[child_i][0].visit_count if child_i in node.children else 0
        for child_i in range(len(node.core_caps))
    )
    visit_counts = torch.tensor(visit_counts, dtype=torch.float)/sum(visit_counts)
    if temp == 0:
      action = torch.argmax(visit_counts).item()
    else:
      probs = torch.softmax(visit_counts/temp, dim=-1)
      action = torch.multinomial(probs, num_samples=1)
    # Return visit counts as they will be used later on during training as logits
    return action, visit_counts


  def __getNewPolicyAndValue(self, node: Node) -> Tuple[torch.Tensor, torch.Tensor]:
    if node.terminal:
      return None, 0
    pol, v_norm = InferenceServer.inference(
      model_name="pred_model",
      unpack=True,
      qubits=self.circuit.alloc_steps[node.allocation_step][(1,2)],
      prev_core_allocs=node.prev_allocs,
      current_core_allocs=node.current_allocs,
      core_capacities=node.core_caps,
      circuit_emb=self.circuit_embs[node.current_slice],
      slice_adj_mat=self.slice_adjm[node.current_slice],
    )
    # Convert normalized value to raw value
    remaining_gates = self.circuit.alloc_steps[node.allocation_step][2]
    v = v_norm*(remaining_gates+1)
    # Add exploration noise to the priors
    dir_noise = torch.distributions.Dirichlet(self.cfg.dirichlet_alpha * torch.ones_like(pol)).sample()
    pol = (1 - self.cfg.noise)*pol + self.cfg.noise*dir_noise
    # Set prior of cores that do not have space for this alloc to zero
    n_qubits = 1 if self.alloc_steps[node.allocation_step,2].item() == -1 else 2
    valid_cores = (node.core_caps >= n_qubits)
    pol[~valid_cores] = 0
    sum_pol = sum_pol
    if sum_pol < 1e-5:
      pol = torch.zeros_like(pol)
      n_valid_cores = sum(valid_cores)
      pol[valid_cores] = 1/n_valid_cores
    else:
      pol /= sum_pol
    return pol, v
  

  def __buildRoot(self) -> Node:
    root = TSPythonEngine.Node(
      current_allocs = -1*torch.ones(size=(self.circuit.n_qubits,), dtype=int),
      prev_allocs = None,
      core_caps = self.core_caps,
      allocation_step = 0,
      current_slice = 0,
      value_sum=0,
    )
    root.policy, root.value_sum = self.__getNewPolicyAndValue(root)
    return root


  def _expandNode(self, node: Node):
    if node.terminal:
      return
    node.children = {}
    qubit0 = self.alloc_steps[node.allocation_step][1].item()
    qubit1 = self.alloc_steps[node.allocation_step][2].item()
    # The prev to terminal node has no next step, but it does have children which contains the cost
    # of each of the actions that can be taken from it
    pre_terminal = (node.allocation_step == self.n_steps-1)
    if not pre_terminal:
      slice_idx_children = self.alloc_steps[node.allocation_step+1,0]

    for action in range(self.n_cores):
      if node.policy[action] < 1e-5:
        continue
      child = TSPythonEngine.Node()
      cost = self.__computeActionCost(node, action)
      node.children[action] = (child, cost)
      if pre_terminal:
        child.terminal = True
        continue

      child.allocation_step = node.allocation_step+1
      child.current_slice = slice_idx_children
      if child.current_slice != node.current_slice:
        child.current_allocs = -1*torch.ones_like(node.current_allocs)
        child.prev_allocs = node.current_allocs.clone()
        child.prev_allocs[qubit0,] = action
        if qubit1 != -1:
          child.prev_allocs[qubit1,] = action
        child.core_caps = self.core_caps
      else:
        child.current_allocs = node.current_allocs.clone()
        child.prev_allocs[qubit0,] = action
        if qubit1 != -1:
          child.prev_allocs[qubit1,] = action
        child.prev_allocs = node.prev_allocs
        child.core_caps = node.core_caps.clone()
        child.core_caps[action] -= len(1 if qubit1 == -1 else 0)
        assert child.core_caps[action] >= 0, 'Not enough space in core to expand'
      child.policy, child.value_sum = self.__getNewPolicyAndValue(child)
  

  def __computeActionCost(self, node: Node, action: int) -> int:
    if node.current_slice == 0:
      return 0
    qubit0 = self.alloc_steps[node.allocation_step][1].item()
    qubit1 = self.alloc_steps[node.allocation_step][2].item()
    qubits = (qubit0,) if qubit1 == -1 else (qubit0, qubit1)
    prev_cores = node.prev_allocs[qubits,]
    costs = self.core_conns[action, prev_cores]
    return torch.sum(costs).item()


  def __backprop(self, search_path: List[Tuple[Node, int]]):
    search_path[-1][0].visit_count += 1
    # Reverse list order and pair items. For example [0,1,2,3] -> ((2,3), (1,2), (0,1))
    for node, next_node in zip(search_path[-2::-1], search_path[:0:-1]):
      node[0].value_sum += next_node[0].value + node[0].children[node[1]]
      node[0].visit_count += 1
  
  
  def __UCB(self, node: Node, action: int) -> float:
    ''' Upper Confidence Bound with minus sign in CB to account for minimization of cost.

    For a nicely formatted version of this formula refer to Appendix B in Ref. [1]
    '''
    return (node.children[action][0].value + node.children[action][1]) - \
           node.policy[action]*sqrt(node.visit_count)/(1+node.children[action][0].visit_count) * \
              (self.cfg.ucb_c1 + log((node.visit_count + self.cfg.ucb_c2 + 1)/self.cfg.ucb_c2))

  
  def __selectChild(self, current_node: Node) -> Tuple[Node, int]:
    (_, action) = min((self.__UCB(current_node, a), a) for a in current_node.children.keys())
    return current_node.children[action], action