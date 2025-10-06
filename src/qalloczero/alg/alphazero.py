import torch
import os
import json
import warnings
from itertools import chain
from enum import Enum
from typing import Tuple, Self, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit, Hardware
from utils.gradient_tools import print_grad_stats, print_grad
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.predmodel import PredictionModel
from qalloczero.alg.ts import TSConfig, TSTrainData, ModelConfigs
from qalloczero.alg.ts_python import TSPythonEngine
from qalloczero.alg.ts_cpp import TSCppEngine
from utils.allocutils import sol_cost
from utils.timer import Timer



class AlphaZero:

  class Backend(Enum):
    Cpp = TSCppEngine
    Python = TSPythonEngine

  @dataclass
  class TrainConfig:
    train_iters: int
    batch_size: int # Number of optimized circuits per batch
    noise_decrease_factor: int
    n_data_augs: int
    sampler: CircuitSampler
    lr: float
    pol_loss_w: float
    ts_cfg: TSConfig
    print_grad_each: Optional[int] = None
    detailed_grad: bool = False


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
      warnings.warn(f"provided folder \"{path}\" already exists")
      if not overwrite:
        i = 2
        while os.path.isdir(path + f"_v{i}"):
          i += 1
        path += f"_v{i}"
        os.makedirs(path)
        warnings.warn(f"Overwrite set to false, saving as \"{path}\"")
      else:
        warnings.warn(f"overwriting previous save file")
    else:
      os.makedirs(path)
    with open(os.path.join(path, "optimizer_conf.json"), "w") as f:
      json.dump(params, f, indent=2)
    torch.save(self.circ_enc.state_dict(), os.path.join(path, "circ_enc.pt"))
    torch.save(self.pred_model.state_dict(), os.path.join(path, "pred_mod.pt"))


  @staticmethod
  def load(path: str, device: str = "cpu") -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "optimizer_conf.json"), "r") as f:
      params = json.load(f)
    # Add backend so that it is possible to load non AlphaZero saved models
    if 'backend' not in params.keys():
      params["backend"] = 'TSCppEngine'
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
    cost = sol_cost(allocations, self.hardware.core_connectivity)
    return allocations, cost, exp_nodes, expl_ratio


  def optimize_mult(
      self,
      circuits: List[Circuit],
      ts_cfg: TSConfig,
  ) -> List[Tuple[torch.Tensor, float, int, float]]:
    circ_adjs = [circ.adj_matrices.to(self.device) for circ in circuits]
    circ_embs = self.circ_enc(torch.stack(circ_adjs))

    def work(azero, **kwargs):
      allocations, exp_nodes, expl_ratio, _ = azero.backend.optimize(**kwargs)
      cost = sol_cost(allocations, azero.hardware.core_connectivity)
      return allocations, cost, exp_nodes, expl_ratio

    with ThreadPoolExecutor(max_workers=len(circuits)) as executor:
      # A dictionary of future : i so that we are able to map results to input circuits later on
      future_map = {
        executor.submit(
          work,
          self,
          core_caps=self.hardware.core_capacities,
          core_conns=self.hardware.core_connectivity,
          slice_adjm=circ_adjs[i],
          circuit_embs=circ_embs[i],
          alloc_steps=circuits[i].alloc_steps,
          cfg=ts_cfg,
          ret_train_data=False,
          verbose=False,
        ) : i
        for i in range(len(circuits))
      }
      results = [(future_map[future], future.result()) for future in as_completed(future_map)]
    return list(map(lambda x: x[1], sorted(results, key=lambda x: x[0])))
  

  def _optimize_mult_train(
    self,
    circuits: List[Circuit],
    adj_mats: torch.Tensor,
    ts_cfg,
  ) -> List[Tuple[float, float, TSTrainData]]:
    adj_mats = adj_mats.to(self.device)
    circ_embs = self.circ_enc(adj_mats)

    def work(azero, **kwargs):
      allocations, exp_nodes, expl_ratio, tdata = azero.backend.optimize(**kwargs)
      norm_cost = sol_cost(allocations, azero.hardware.core_connectivity)/kwargs['alloc_steps'][0][3]
      return norm_cost, expl_ratio, tdata

    with ThreadPoolExecutor(max_workers=len(adj_mats)) as executor:
      # A dictionary of future : i so that we are able to map results to input circuits later on
      future_map = {
        executor.submit(
          work,
          self,
          core_caps=self.hardware.core_capacities,
          core_conns=self.hardware.core_connectivity,
          slice_adjm=adj_mats[i],
          circuit_embs=circ_embs[i],
          alloc_steps=circuits[i].alloc_steps,
          cfg=ts_cfg,
          ret_train_data=True,
          verbose=False,
        ) : i
        for i in range(len(circuits))
      }
      results = [(future_map[future], future.result()) for future in as_completed(future_map)]
    return list(map(lambda x: x[1], sorted(results, key=lambda x: x[0])))


  def train(
    self,
    train_cfg: TrainConfig,
    train_device: str = '',
  ):
    if not train_device:
      train_device = self.device
    self.timer = Timer.get("_optimizer_timer")
    self.iter_timer = Timer.get("_train_iter_timer")
    self.timer.reset()
    self.iter_timer.reset()
    prob_loss = torch.nn.CrossEntropyLoss()
    val_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
      chain(self.circ_enc.parameters(), self.pred_model.parameters()), lr=train_cfg.lr
    )
    self.pgrad_counter = 1

    try:
      for iter in range(train_cfg.train_iters):
        self.iter_timer.start()
        print(f"[*] Train iter {iter+1}/{train_cfg.train_iters}")
        self.circ_enc.eval()
        self.circ_enc.to(self.device)
        self.pred_model.eval()
        self.pred_model.to(self.device)
        circuits = [train_cfg.sampler.sample() for _ in range(train_cfg.batch_size)]
        circ_adjs = torch.stack([circ.adj_matrices for circ in circuits])
        avg_cost = 0
        avg_expl_r = 0
        train_data = []
        with self.timer:
          results = self._optimize_mult_train(circuits, circ_adjs, train_cfg.ts_cfg)
        for (cost, er, tdata) in results:
          avg_cost += cost
          avg_expl_r += er
          train_data.append(tdata)
        print((
          f" + Obtained train data: t={self.timer.time:.2f} ac={avg_cost/train_cfg.batch_size:.3f} "
          f" er={avg_expl_r/train_cfg.batch_size:.3f} (noise={train_cfg.ts_cfg.noise:.3f})"
        ))
        
        avg_loss = 0
        avg_loss_pol = 0
        avg_loss_val = 0
        circ_adjs = circ_adjs.to(train_device)
        core_cons = self.hardware.core_connectivity.to(train_device)
        with self.timer:
          self.circ_enc.train()
          self.circ_enc.to(train_device)
          self.pred_model.train()
          self.pred_model.output_logits(True)
          self.pred_model.to(train_device)
          for batch, tdata in enumerate(train_data):
            qubits, prev_allocs, curr_allocs, core_caps, slice_idx, ref_logits, ref_values = self._move_train_data(tdata, train_device)
            for _ in range(train_cfg.n_data_augs):
              perm, perm_qubits = self._perm_qubits(qubits, train_device)
              new_adj = circ_adjs[batch][:,perm,:][:,:,perm]
              circ_embs = self.circ_enc(new_adj.unsqueeze(0)).squeeze(0)
              pols, values = self.pred_model(
                qubits=perm_qubits,
                prev_core_allocs=prev_allocs[:,perm],
                current_core_allocs=curr_allocs[:,perm],
                core_capacities=core_caps,
                core_connectivity=core_cons,
                circuit_emb=circ_embs[slice_idx],
                slice_adj_mat=new_adj[slice_idx],
              )
              loss_pols = train_cfg.pol_loss_w * prob_loss(pols, ref_logits)
              loss_vals = (1 - train_cfg.pol_loss_w) * val_loss(values, ref_values)
              loss = loss_pols + loss_vals
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              avg_loss += loss
              avg_loss_pol += loss_pols
              avg_loss_val += loss_vals
        self.pred_model.output_logits(False)
        n_iters = train_cfg.batch_size*train_cfg.n_data_augs
        print((
          f" + Updated models: t={self.timer.time:.2f} loss={avg_loss/n_iters:.4f} "
          f" (pol={avg_loss_pol/n_iters:.4f}, val={avg_loss_val/n_iters:.4f})"
        ))

        train_cfg.ts_cfg.noise *= train_cfg.noise_decrease_factor
        self.backend.replace_model(self.pred_model)
        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - iter - 1)
        print((
          f" + t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left)"
        ))
        if train_cfg.print_grad_each is not None and self.pgrad_counter == train_cfg.print_grad_each:
          if train_cfg.detailed_grad:
            print(f"\n[+] Gradient information for prediction model:")
            print_grad(self.pred_model)
            print(f"\n[+] Gradient information for circuit encoder:")
            print_grad(self.circ_enc)
          else:
            print_grad_stats(self.pred_model, 'prediction model')
            print_grad_stats(self.circ_enc, 'circuit encoder')
          self.pgrad_counter = 1
        else:
          self.pgrad_counter += 1
    except KeyboardInterrupt as e:
      if 'y' not in input('\nGraceful shutdown? [y/n]: ').lower():
        raise e
  

  def _move_train_data(self, tdata, train_device) -> Tuple[torch.Tensor, ...]:
    qubits = tdata.qubits.to(train_device)
    prev_allocs = tdata.prev_allocs.to(train_device)
    curr_allocs = tdata.curr_allocs.to(train_device)
    core_caps = tdata.core_caps.to(train_device)
    slice_idx = tdata.slice_idx.to(train_device).flatten()
    ref_logits = tdata.logits.to(train_device)
    ref_values = tdata.value.to(train_device)
    return qubits, prev_allocs, curr_allocs, core_caps, slice_idx, ref_logits, ref_values
  

  def _perm_qubits(self, qubits, device) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(self.hardware.n_qubits, dtype=torch.int, device=device)
    qubit_mask = qubits != -1
    perm_qubits = qubits.clone()
    perm_qubits[qubit_mask] = perm[qubits[qubit_mask]]
    return perm, perm_qubits