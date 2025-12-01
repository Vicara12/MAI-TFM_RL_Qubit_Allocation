import torch
import os
import json
import warnings
from time import time
from enum import Enum
from typing import Tuple, Self, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from sampler.hardwaresampler import HardwareSampler
from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit, Hardware
from utils.gradient_tools import print_grad_stats, print_grad
from qalloczero.models.predmodel import PredictionModel
from qalloczero.alg.ts import TSConfig, TSTrainData, ModelConfigs
from qalloczero.alg.ts_python import TSPythonEngine
from qalloczero.alg.ts_cpp import TSCppEngine
from utils.allocutils import sol_cost, get_all_checkpoints
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
    circ_sampler: CircuitSampler
    lr: float
    ts_cfg: TSConfig
    hardware_sampler: HardwareSampler
    minimum_noise: float = 0.2
    print_grad_each: Optional[int] = None
    detailed_grad: bool = False


  def __init__(
    self,
    device: str = "cpu",
    backend: Backend = Backend.Cpp,
    model_cfg: ModelConfigs = ModelConfigs()
  ):
    self.device = device
    self.model_cfg = model_cfg
    self.backend = backend.value(
      device=device,
    )
    self.pred_model = PredictionModel(layers=model_cfg.layers)
    self.backend.load_model(self.pred_model)
    self.pred_model.to(device)


  def save(self, path: str, overwrite: bool = False):
    params = dict(
      backend=self.backend.__class__.__name__,
      layers=self.model_cfg.layers,
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
    torch.save(self.pred_model.state_dict(), os.path.join(path, "pred_mod.pt"))
    return path


  @staticmethod
  def load(path: str, device: str = "cpu", checkpoint: Optional[int] = None) -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "optimizer_conf.json"), "r") as f:
      params = json.load(f)
    # Add backend so that it is possible to load non AlphaZero saved models
    if 'backend' not in params.keys():
      params["backend"] = 'TSCppEngine'
    model_cfg = ModelConfigs(layers=params['layers'])
    backend = AlphaZero.Backend.Cpp if params["backend"] == 'TSCppEngine' else AlphaZero.Backend.Python
    loaded = AlphaZero(device=device, backend=backend, model_cfg=model_cfg)
    model_file = "pred_mod.pt"
    if checkpoint is not None:
      chpt_files = get_all_checkpoints(path)
      if checkpoint not in chpt_files.keys():
        raise Exception(f'Checkpoint {checkpoint} not found: {", ".join(list(chpt_files.keys()))}')
      model_file = chpt_files[checkpoint]
    loaded.pred_model.load_state_dict(
      torch.load(
        os.path.join(path, model_file),
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
    hardware: Hardware,
    verbose: bool = False,
  ) -> Tuple[torch.Tensor, float, int, float]:
    if circuit.n_qubits != hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {hardware.n_qubits} != {circuit.n_qubits}"
      ))
    allocations, exp_nodes, expl_ratio, _ = self.backend.optimize(
      n_qubits=circuit.n_qubits,
      core_caps=hardware.core_capacities,
      core_conns=hardware.core_connectivity,
      circuit_embs=circuit.embedding,
      alloc_steps=circuit.alloc_steps,
      cfg=ts_cfg,
      ret_train_data=False,
      verbose=verbose,
    )
    cost = sol_cost(allocations, hardware.core_connectivity)
    return allocations, cost, exp_nodes, expl_ratio


  def optimize_mult(
      self,
      circuits: List[Circuit],
      ts_cfg: TSConfig,
      hardware: Hardware,
  ) -> List[Tuple[torch.Tensor, float, int, float]]:
    def work(azero, **kwargs):
      allocations, exp_nodes, expl_ratio, _ = azero.backend.optimize(**kwargs)
      cost = sol_cost(allocations, hardware.core_connectivity)
      return allocations, cost, exp_nodes, expl_ratio

    with ThreadPoolExecutor(max_workers=len(circuits)) as executor:
      # A dictionary of future : i so that we are able to map results to input circuits later on
      future_map = {
        executor.submit(
          work,
          self,
          n_qubits=circuits[i].n_qubits,
          core_caps=hardware.core_capacities,
          core_conns=hardware.core_connectivity,
          circuit_embs=circuits[i].embedding,
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
    hardware: Hardware,
    ts_cfg,
  ) -> List[Tuple[float, float, TSTrainData]]:
    def work(azero, **kwargs):
      allocations, exp_nodes, expl_ratio, tdata = azero.backend.optimize(**kwargs)
      norm_cost = sol_cost(allocations, hardware.core_connectivity)/kwargs['alloc_steps'][0][3]
      return norm_cost, expl_ratio, tdata
    
    with ThreadPoolExecutor(max_workers=len(circuits)) as executor:
      # A dictionary of future : i so that we are able to map results to input circuits later on
      future_map = {
        executor.submit(
          work,
          self,
          n_qubits=circuits[i].n_qubits,
          core_caps=hardware.core_capacities,
          core_conns=hardware.core_connectivity,
          circuit_embs=circuits[i].embedding,
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
  ) -> dict[str, list]:
    if not train_device:
      train_device = self.device
    self.timer = Timer.get("_optimizer_timer")
    self.iter_timer = Timer.get("_train_iter_timer")
    self.timer.reset()
    self.iter_timer.reset()
    prob_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
      self.pred_model.parameters(), lr=train_cfg.lr
    )
    self.pgrad_counter = 1
    data_log = dict(
      loss = [],
      noise = [],
      cost = [],
      t = []
    )

    try:
      for iter in range(train_cfg.train_iters):
        hardware = train_cfg.hardware_sampler.sample()
        print(f"Hardware: {hardware.core_capacities.tolist()} nq={hardware.n_qubits} nc={hardware.n_cores}")
        train_cfg.circ_sampler.num_lq = hardware.n_qubits
        self.iter_timer.start()
        print(f"[*] Train iter {iter+1}/{train_cfg.train_iters}")
        self.pred_model.eval()
        self.pred_model.to(self.device)
        circuits = [train_cfg.circ_sampler.sample()]*train_cfg.batch_size
        costs = []
        avg_expl_r = 0
        train_data = []
        with self.timer:
          results = self._optimize_mult_train(circuits, hardware, train_cfg.ts_cfg)
        for (cost, er, tdata) in results:
          costs.append(cost)
          avg_expl_r += er
          train_data.append(tdata)
        costs = torch.tensor(costs)
        print((
          f" + Obtained train data: t={self.timer.time:.2f} ac={costs.mean().item():.3f} ({costs.std(unbiased=True).item():.2f})"
          f" er={avg_expl_r/train_cfg.batch_size:.3f} (noise={train_cfg.ts_cfg.noise:.3f})"
        ))
        costs = (costs - costs.mean()) / (costs.std(unbiased=True) + 1e-8)
        
        avg_loss = 0
        core_cons = hardware.core_connectivity.to(train_device)
        circ_emb = circuits[0].embedding.to(train_device)
        with self.timer:
          self.pred_model.train()
          self.pred_model.output_logits(True)
          self.pred_model.to(train_device)
          for batch, tdata in enumerate(train_data):
            qubits, prev_allocs, curr_allocs, core_caps, slice_idx, ref_logits, _ = self._move_train_data(tdata, train_device)
            pols, _, _ = self.pred_model(
              qubits=qubits,
              prev_core_allocs=prev_allocs,
              current_core_allocs=curr_allocs,
              core_capacities=core_caps,
              core_connectivity=core_cons,
              circuit_emb=circ_emb[slice_idx],
            )
            # weight = 1 if costs[batch] == 0 else 1/costs[batch] # Protection against Nans
            loss = costs[batch]*prob_loss(pols, ref_logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_norm=1)
            optimizer.step()
            avg_loss += loss
        self.pred_model.output_logits(False)
        n_iters = train_cfg.batch_size*train_cfg.n_data_augs
        print((
          f" + Updated models: t={self.timer.time:.2f} loss={avg_loss/n_iters:.4f}"
        ))

        data_log['loss'].append(avg_loss.item())
        data_log['noise'].append(train_cfg.ts_cfg.noise)
        data_log['cost'].append((sum(costs)/len(costs)).item())
        data_log['t'].append(time())

        train_cfg.ts_cfg.noise *= train_cfg.noise_decrease_factor
        train_cfg.ts_cfg.noise = max(train_cfg.ts_cfg.noise, train_cfg.minimum_noise)
        self.backend.replace_model(self.pred_model)
        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - iter - 1)
        print((
          f" + t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left)"
        ))
        if train_cfg.print_grad_each is not None and self.pgrad_counter == train_cfg.print_grad_each:
          if train_cfg.detailed_grad:
            print_grad(self.pred_model)
          else:
            print_grad_stats(self.pred_model, 'prediction model')
          self.pgrad_counter = 1
        else:
          self.pgrad_counter += 1

    except KeyboardInterrupt as e:
      if 'y' not in input('\nGraceful shutdown? [y/n]: ').lower():
        raise e
    
    return data_log
  

  def _move_train_data(self, tdata, train_device) -> Tuple[torch.Tensor, ...]:
    qubits = tdata.qubits.to(train_device)
    prev_allocs = tdata.prev_allocs.to(train_device)
    curr_allocs = tdata.curr_allocs.to(train_device)
    core_caps = tdata.core_caps.to(train_device)
    slice_idx = tdata.slice_idx.to(train_device).flatten()
    ref_logits = tdata.logits.to(train_device)
    ref_values = tdata.value.to(train_device)
    return qubits, prev_allocs, curr_allocs, core_caps, slice_idx, ref_logits, ref_values