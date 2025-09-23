import torch
from typing import Tuple, List, Union
from dataclasses import dataclass
from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit, Hardware, GateType
from qalloczero.alg.mcts import MCTS
from qalloczero.models.inferenceserver import InferenceServer
from utils.environment import QubitAllocationEnvironment
from utils.allocutils import solutionCost
from utils.timer import Timer



class AlphaZero:

  @dataclass
  class Config:
    hardware: Hardware
    # First item of encoder_shape determines qubit embedding size and last item circuit emb. size
    encoder_shape: Tuple[int]
    mcts_config: MCTS.Config
    

  @dataclass
  class TrainConfig:
    train_iters: int
    batch_size: int # Number of optimized circuits per batch
    sampler: CircuitSampler
    lr: float
    v_weight: float
    logit_weight: float


  @dataclass
  class TrainDataItem:
    qubits: Union[GateType, Tuple[int]]
    current_alloc: torch.Tensor
    prev_alloc: torch.Tensor
    core_caps: torch.Tensor
    curr_slice: int
    logits: torch.Tensor
    value: float



  def __init__(self, config: Config, qubit_embs: torch.Tensor):
    assert InferenceServer.hasModel('circ_enc')
    self.cfg = config
    self.optimizing_for_train = False
  

  def _initMCTS(self, circuit: Circuit) -> MCTS:
    (circuit_embs, slice_embs) = InferenceServer.inference(model_name='circ_enc', unpack=True, circuits=[circuit])
    return MCTS(
      slice_embs=slice_embs,
      circuit_embs=circuit_embs,
      circuit=circuit,
      hardware=self.cfg.hardware,
      config=self.cfg.mcts_config
    )


  def optimize(self, circuit: Circuit) -> Tuple[torch.Tensor, float]:
    ''' Returns tensor with qubit allocations and exploration ratio.
    '''
    env = QubitAllocationEnvironment(circuit=circuit, hardware=self.cfg.hardware)
    mcts = self._initMCTS(circuit=circuit)

    # Run MCTS
    n_expanded_nodes = 0
    for step_i, alloc_step in enumerate(circuit.alloc_steps):
      (_, qubits_step) = alloc_step
      alloc_to_core, _, n_sims = mcts.iterate()
      n_expanded_nodes += n_sims
      total_cost = 0
      for qubit in qubits_step:
        total_cost += env.allocate(alloc_to_core, qubit)
      print((f" [{step_i+1}/{circuit.n_steps} "
             f"slc={alloc_step[0]} {alloc_step[1]} -> {alloc_to_core}] "
             f"sims={n_sims} cost={total_cost}"))
    
    n_cores = self.cfg.hardware.n_cores
    theoretical_n_expanded_nodes = circuit.n_steps * \
      self.cfg.mcts_config.target_tree_size * (n_cores - 1)/n_cores

    return env.qubit_allocations, n_expanded_nodes/theoretical_n_expanded_nodes
  

  def _optimizeTrain(self, circuit: Circuit) -> Tuple[torch.Tensor, List]:
    env = QubitAllocationEnvironment(circuit=circuit, hardware=self.cfg.hardware)
    mcts = self._initMCTS(circuit=circuit)
    train_data = []

    # Run MCTS
    for step_i, alloc_step in enumerate(circuit.alloc_steps):
      root = mcts.root
      train_data.append(AlphaZero.TrainDataItem(
        qubits=alloc_step[1],
        current_alloc=root.current_allocs,
        prev_alloc=root.prev_allocs,
        core_caps=root.core_caps,
        curr_slice=root.current_slice,
        final_logits=None, # Post-MCTS logits of action priors (core alloc)
        final_value=0     # Final value (allocation cost) prediction for this node
      ))
      (_, qubits_step, _) = alloc_step
      alloc_to_core, logits, n_sims = mcts.iterate()
      train_data[-1].final_logits = logits
      for qubit in qubits_step:
        train_data[-1].final_value += env.allocate(alloc_to_core, qubit)
    
    # Compute V for each action i by adding the total allocation cost from i until the end
    # Iterate the list of actions backwards ignoring the last item
    for i in range(len(train_data)-2,-1,-1):
      train_data[i].final_value += train_data[i+1].final_value
    # Now go through the list again normalizing values
    for i in range(circuit.n_steps):
      remaining_gates = circuit.alloc_steps[i][2]
      train_data[i].final_value /= (remaining_gates+1)
    return env.qubit_allocations, train_data


  def _updateModels(self, circuits: List[Circuit], train_data: List, train_cfg: TrainConfig) -> None:
    circ_enc = InferenceServer.model('circ_enc')
    pred_model = InferenceServer.model("pred_model")
    snap_enc = InferenceServer.model("snap_enc")

    circ_enc.train()
    pred_model.train()
    snap_enc.train()

    # All three models share the qubit embeddings, which is problematic (it will lead to a repeated
    # parameter warning). This creates a set of parameters without repeated instances.
    unique_params = list({id(p): p for p in (
        list(circ_enc.parameters()) +
        list(pred_model.parameters()) +
        list(snap_enc.parameters())
    )}.values())

    optim = torch.optim.Adam(unique_params, lr=train_cfg.lr)
    optim.zero_grad()

    logit_crit = torch.nn.CrossEntropyLoss()
    value_crit = torch.nn.MSELoss()
    loss = 0

    embs = InferenceServer.inference(model_name='circ_enc', unpack=False, circuits=circuits)

    for i, (emb, tdata) in enumerate(zip(embs, train_data)):
      (circ_emb, slice_emb) = emb
      prev_core_allocs = None
      empty_core_mat = -1*torch.ones(size=(circuits[i].n_qubits,), dtype=int)
      core_embs = InferenceServer.inference(model_name="snap_enc", unpack=True, core_allocs=[empty_core_mat])
      total_logit_loss = 0
      total_value_loss = 0
      for sample in tdata:
        if prev_core_allocs is not None and prev_core_allocs != sample.prev_alloc:
          prev_core_allocs = sample.prev_alloc
          core_embs = InferenceServer.inference(model_name='snap_enc', unpack=True, core_allocs=[prev_core_allocs])

        _, v, logits = InferenceServer.inference(
          model_name='pred_model', unpack=False,
          qubits=sample.qubits,
          core_embs=core_embs,
          prev_core_allocs=sample.prev_alloc,
          current_core_capacities=sample.core_caps,
          circuit_emb=circ_emb[sample.curr_slice],
          slice_emb=slice_emb[sample.curr_slice]
        )

        logit_loss = train_cfg.logit_weight * logit_crit(logits, sample.final_logits)
        value_loss = train_cfg.v_weight * value_crit(v, torch.tensor([sample.final_value], dtype=torch.float))
        loss += logit_loss + value_loss
        total_logit_loss += logit_loss.item()
        total_value_loss += value_loss.item()
      print(f"  - optim_{i} logit_loss={total_logit_loss/len(tdata):.6f} v_loss={total_value_loss/len(tdata):.6f}")
    
    loss.backward()
    optim.step()

    circ_enc.eval()
    pred_model.eval()
    snap_enc.eval()


  def train(self, train_cfg: TrainConfig) -> None:
    t = Timer.get("optimize_circuit_train")
    for train_i in range(train_cfg.train_iters):
      circuits = []
      train_data_all = []
      print(f"\n[*] Loop: {train_i}/{train_cfg.train_iters}")
      for batch_i in range(train_cfg.batch_size):
        circuits.append(train_cfg.sampler.sample())
        with t:
          allocs, train_data = self._optimizeTrain(circuit=circuits[-1])
        train_data_all.append(train_data)
        sol_cost = solutionCost(allocs, self.cfg.hardware.core_connectivity)
        print((
          f"  - batch_{batch_i}  "
          f"cost={sol_cost}  "
          f"gates={circuits[-1].n_gates} ({sol_cost/circuits[-1].n_gates:.2f})  "
          f"time={t.total_time:.2f}s "
        ))
        t.reset()
      print("   <Training>")
      self._updateModels(circuits=circuits, train_data=train_data_all, train_cfg=train_cfg)