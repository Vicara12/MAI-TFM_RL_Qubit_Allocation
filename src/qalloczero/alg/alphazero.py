import torch
import os
import json
from itertools import chain
from enum import Enum
from typing import Tuple, Self
from dataclasses import dataclass
from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit, Hardware
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.predmodel import PredictionModel
from qalloczero.alg.ts import TSConfig, TSTrainData
from qalloczero.alg.ts_python import TSPythonEngine
from qalloczero.alg.ts_cpp import TSCppEngine
from utils.allocutils import solutionCost
from utils.timer import Timer



class AlphaZero:

  class Backend(Enum):
    Cpp = TSCppEngine
    Python = TSPythonEngine
  
  @dataclass
  class ModelConfigs:
    ce_nheads: int = 4
    ce_nlayers: int = 4
    pm_nemb_sz: int = 4
    pm_nheads: int = 4

  @dataclass
  class TrainConfig:
    train_iters: int
    batch_size: int # Number of optimized circuits per batch
    n_data_augs: int
    sampler: CircuitSampler
    lr: float
    pol_loss_w: float
    ts_cfg: TSConfig


  def __init__(
    self,
    hardware: Hardware,
    device: str = "cpu",
    backend: Backend = Backend.Cpp,
    model_cfg: ModelConfigs = ModelConfigs()
  ):
    self.hardware = hardware
    self.device = device
    self.backend = backend.value(
      n_qubits=hardware.n_qubits,
      n_cores=hardware.n_cores,
      device=device,
    )
    self.circ_enc = CircuitEncoder(
      n_qubits=hardware.n_qubits,
      n_heads=model_cfg.ce_nheads,
      n_layers=model_cfg.ce_nlayers,
    )
    self.circ_enc.to(device)
    self.pred_model = PredictionModel(
      n_qubits=hardware.n_qubits,
      n_cores=hardware.n_cores,
      number_emb_size=model_cfg.pm_nemb_sz,
      n_heads=model_cfg.ce_nheads,
    )
    self.backend.load_model(self.pred_model)
    self.pred_model.to(device)


  def save(self, path: str, overwrite: bool = False):
    params = dict(
      core_caps=self.hardware.core_capacities.tolist(),
      core_conns=self.hardware.core_connectivity.tolist(),
      backend=self.backend.__class__.__name__
    )
    if os.path.isdir(path):
      if not overwrite:
        raise Exception(f"Provided save directory already exists: {path}")
    else:
      os.makedirs(path)
    with open(os.path.join(path, "azero.json"), "w") as f:
      json.dump(params, f, indent=2)
    torch.save(self.circ_enc.state_dict(), os.path.join(path, "circ_enc.pt"))
    torch.save(self.pred_model.state_dict(), os.path.join(path, "pred_mod.pt"))


  @staticmethod
  def load(path: str, device: str = "cpu") -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "azero.json"), "r") as f:
      params = json.load(f)
    hardware = Hardware(torch.tensor(params["core_caps"]), torch.tensor(params["core_conns"]))
    backend = AlphaZero.Backend.Cpp if params["backend"] == 'TSCppEngine' else AlphaZero.Backend.Python
    loaded = AlphaZero(hardware=hardware, device=device, backend=backend)
    loaded.circ_enc.load_state_dict(
      torch.load(
        os.path.join(path, "circ_enc.pt"),
        weights_only=False,
        map_location=device,
      )
    )
    loaded.pred_model.load_state_dict(
      torch.load(
        os.path.join(path, "pred_mod.pt"),
        weights_only=False,
        map_location=device,
      )
    )
    loaded.backend.replace_model(loaded.pred_model)
    return loaded
    

  def optimize(
    self,
    circuit: Circuit,
    ts_cfg: TSConfig,
    verbose: bool = False,
  ) -> Tuple[torch.Tensor, float, int, float]:
    if circuit.n_qubits != self.hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {self.hardware.n_qubits} != {circuit.n_qubits}"
      ))
    circ_embs = self.circ_enc(circuit.adj_matrices.unsqueeze(0).to(self.device)).squeeze(0)
    allocations, exp_nodes, expl_ratio, _ = self.backend.optimize(
      core_caps=self.hardware.core_capacities,
      core_conns=self.hardware.core_connectivity,
      slice_adjm=circuit.adj_matrices,
      circuit_embs=circ_embs,
      alloc_steps=circuit.alloc_steps,
      cfg=ts_cfg,
      ret_train_data=False,
      verbose=verbose,
    )
    cost = solutionCost(allocations, self.hardware.core_connectivity)
    return allocations, cost, exp_nodes, expl_ratio
  

  def _optimize_train(
      self,
      slice_adjm: torch.Tensor,
      circ_embs: torch.Tensor,
      alloc_steps: torch.Tensor,
      cfg: TSConfig,
  ) -> Tuple[float, float, float, TSTrainData]:
    with self.timer:
      allocations, _, expl_ratio, train_data = self.backend.optimize(
        core_caps=self.hardware.core_capacities,
        core_conns=self.hardware.core_connectivity,
        slice_adjm=slice_adjm,
        circuit_embs=circ_embs,
        alloc_steps=alloc_steps,
        cfg=cfg,
        ret_train_data=True,
        verbose=False,
      )
    cost = solutionCost(allocations, self.hardware.core_connectivity)
    return cost, self.timer.time, expl_ratio, train_data
  

  def train(
    self,
    train_cfg: TrainConfig,
  ):
    self.timer = Timer.get("_optimizer_timer")
    self.iter_timer = Timer.get("_train_iter_timer")
    prob_loss = torch.nn.CrossEntropyLoss()
    val_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
      chain(self.circ_enc.parameters(), self.pred_model.parameters()), lr=train_cfg.lr
    )

    for iter in range(train_cfg.train_iters):
      self.iter_timer.start()
      print(f"[*] Train iter {iter+1}/{train_cfg.train_iters}")
      self.circ_enc.eval()
      self.pred_model.eval()
      circuits = [train_cfg.sampler.sample() for _ in range(train_cfg.batch_size)]
      circ_adjs = [circ.adj_matrices.to(self.device) for circ in circuits]
      circ_embs = self.circ_enc(torch.stack(circ_adjs))
      if torch.any(torch.isnan(circ_embs)):
        pass
      train_data = []
      for batch in range(train_cfg.batch_size):
        cost, time, expl_ratio, train_data_i = self._optimize_train(
          slice_adjm=circ_adjs[batch],
          circ_embs=circ_embs[batch],
          alloc_steps=circuits[batch].alloc_steps,
          cfg=train_cfg.ts_cfg,
        )
        train_data.append(train_data_i)
        print((
          f" - [{batch+1}/{train_cfg.batch_size}] t={time:.2f} c={cost:.1f} "
          f"({cost/circuits[batch].n_gates:.2f}) er={expl_ratio:.3f} "
        ))
      
      avg_loss = 0
      avg_loss_pol = 0
      avg_loss_val = 0
      core_cons = self.hardware.core_connectivity.to(self.device)
      with self.timer:
        self.circ_enc.train()
        self.pred_model.train()
        for _ in range(train_cfg.n_data_augs):
          perm = torch.randperm(self.hardware.n_qubits, dtype=torch.int, device=self.device)
          for batch, tdata in enumerate(train_data):
            new_adj = circ_adjs[batch][:,perm,:][:,:,perm]
            qubit_mask = tdata.qubits != -1
            new_qubits = tdata.qubits.clone()
            new_qubits[qubit_mask] = perm[tdata.qubits[qubit_mask]]
            circ_embs = self.circ_enc(new_adj.unsqueeze(0)).squeeze(0)
            pols, values = self.pred_model(
              qubits=new_qubits,
              prev_core_allocs=tdata.prev_allocs[:,perm],
              current_core_allocs=tdata.curr_allocs[:,perm],
              core_capacities=tdata.core_caps,
              core_connectivity=core_cons,
              circuit_emb=circ_embs[tdata.slice_idx.flatten()],
              slice_adj_mat=new_adj[tdata.slice_idx.flatten()],
            )
            loss_pols = train_cfg.pol_loss_w * prob_loss(pols, tdata.logits)
            loss_vals = (1 - train_cfg.pol_loss_w) * val_loss(values, tdata.value)
            loss = loss_pols + loss_vals
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss
            avg_loss_pol += loss_pols
            avg_loss_val += loss_vals
      n_iters = train_cfg.batch_size*train_cfg.n_data_augs
      print((
        f" + Updated models: t={self.timer.time:.2f} loss={avg_loss/n_iters:.4f} "
        f" (pol={avg_loss_pol/n_iters:.4f}, val={avg_loss_val/n_iters:.4f})"
      ))

      self.backend.replace_model(self.pred_model)
      self.iter_timer.stop()
      t_left = self.iter_timer.avg_time*(train_cfg.train_iters - iter - 1)
      print((
        f" + t={self.iter_timer.time:.2f}s "
        f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left)"
      ))