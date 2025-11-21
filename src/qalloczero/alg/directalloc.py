import os
import json
import torch
import warnings
from time import time
from copy import deepcopy
from typing import Self, Tuple, Optional
from dataclasses import dataclass
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost
from scipy.stats import ttest_ind
from utils.timer import Timer
from sampler.hardwaresampler import HardwareSampler
from sampler.circuitsampler import CircuitSampler
from qalloczero.alg.ts import ModelConfigs
from qalloczero.models.predmodel import PredictionModel



@dataclass
class DAConfig:
  noise: float = 0.0
  mask_invalid: bool = True
  greedy: bool = True


class DirectAllocator:

  @dataclass
  class TrainConfig:
    train_iters: int
    batch_size: int
    group_size: int
    validate_each: int
    validation_hardware: Hardware
    validation_circuits: list[Circuit]
    store_path: str
    initial_noise: float
    noise_decrease_factor: int
    circ_sampler: CircuitSampler
    lr: float
    hardware_sampler: HardwareSampler


  def __init__(
    self,
    device: str = "cpu",
    model_cfg: ModelConfigs = ModelConfigs()
  ):
    self.model_cfg = model_cfg
    self.pred_model = PredictionModel(layers=model_cfg.layers)
    self.pred_model.to(device)
  

  @property
  def device(self) -> torch.device:
    return next(self.pred_model.parameters()).device


  def _save_model_cfg(self, path: str):
    params = dict(
      layers=self.model_cfg.layers,
    )
    with open(os.path.join(path, "optimizer_conf.json"), "w") as f:
      json.dump(params, f, indent=2)


  def _make_save_dir(self, path: str, overwrite: bool) -> str:
    if os.path.isdir(path):
      if not overwrite:
        i = 2
        while os.path.isdir(path + f"_v{i}"):
          i += 1
        path += f"_v{i}"
        os.makedirs(path)
        warnings.warn(f"Provided folder \"{path}\" already exists, saving as \"{path}\"")
      else:
        warnings.warn(f"Provided folder \"{path}\" already exists, overwriting previous save file")
    else:
      os.makedirs(path)
    self._save_model_cfg(path)
    return path


  def save(self, path: str, overwrite: bool = False):
    path = self._make_save_dir(path=path, overwrite=overwrite)
    torch.save(self.pred_model.state_dict(), os.path.join(path, "pred_mod.pt"))
    return path


  @staticmethod
  def load(path: str, device: str = "cuda") -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "optimizer_conf.json"), "r") as f:
      params = json.load(f)
    model_cfg = ModelConfigs(layers=params['layers'])
    loaded = DirectAllocator(device=device, model_cfg=model_cfg)
    loaded.pred_model.load_state_dict(
      torch.load(
        os.path.join(path, "pred_mod.pt"),
        weights_only=False,
        map_location=device,
      )
    )
    return loaded
  

  def _sample_action(
    self,
    logits: torch.Tensor,
    core_caps: torch.Tensor,
    n_qubits: int,
    cfg: DAConfig,
  ) -> Tuple[int, int]:
    # Set prior of cores that do not have space for this alloc to zero
    valid_cores = (core_caps >= n_qubits).expand(logits.shape).reshape((-1,))
    pol = torch.softmax(logits.reshape((-1,)), dim=-1)
    assert valid_cores.any().item(), "No valid allocation possible"
    if cfg.mask_invalid:
      pol[~valid_cores] = 0
    # Add exploration noise to the priors
    if cfg.noise != 0:
      noise = torch.abs(torch.randn(pol.shape, device=self.device))
      noise[~valid_cores] = 0
      pol = (1 - cfg.noise)*pol + cfg.noise*noise
    sum_pol = pol.sum()
    if cfg.mask_invalid and sum_pol < 1e-5:
      pol = torch.zeros_like(pol)
      pol[valid_cores] = 1/sum(valid_cores)
    else:
      pol /= sum_pol
    action = pol.argmax() if cfg.greedy else torch.distributions.Categorical(pol).sample()
    (qubit_set, core) = torch.unravel_index(action, logits.shape)
    return qubit_set.item(), core.item()


  def _allocate(
    self,
    allocations: torch.Tensor,
    circ_embs: torch.Tensor,
    alloc_slices: list[tuple[int, list[int], list[tuple[int,int]]]],
    cfg: DAConfig,
    hardware: Hardware,
    ret_train_data: bool,
  ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    core_caps_orig = hardware.core_capacities.to(self.device)
    core_allocs = hardware.n_cores * torch.ones(
      [hardware.n_qubits],
      dtype=torch.long,
      device=self.device,
    )
    prev_core_allocs = None
    core_caps = None
    if ret_train_data:
      all_log_probs = []
    dev_core_con = hardware.core_connectivity.to(self.device)
    for slice_idx, (_, free_qubits, paired_qubits) in enumerate(alloc_slices):
      prev_core_allocs = core_allocs
      core_allocs = hardware.n_cores * torch.ones_like(core_allocs)
      core_caps = core_caps_orig.clone()
      paired_qubits = list(paired_qubits)
      free_qubits = list(free_qubits)
      
      while paired_qubits:
        _, _, log_pol = self.pred_model(
          qubits=torch.tensor(paired_qubits, dtype=torch.int, device=self.device),
          prev_core_allocs=prev_core_allocs.expand((len(paired_qubits), len(prev_core_allocs))),
          current_core_allocs=core_allocs.expand((len(paired_qubits), len(prev_core_allocs))),
          core_capacities=core_caps.expand((len(paired_qubits), hardware.n_cores)),
          core_connectivity=dev_core_con,
          circuit_emb=circ_embs[:,slice_idx,:,:].expand((len(paired_qubits), -1, -1)),
        )
        qubit_set, core = self._sample_action(
          logits=log_pol,
          core_caps=core_caps,
          n_qubits=2,
          cfg=cfg
        )
        allocations[slice_idx,paired_qubits[qubit_set][0]] = core
        core_allocs[paired_qubits[qubit_set][0]] = core
        allocations[slice_idx,paired_qubits[qubit_set][1]] = core
        core_allocs[paired_qubits[qubit_set][1]] = core
        if ret_train_data:
          all_log_probs.append(log_pol[qubit_set, core])
        core_caps[core] -= 2
        if cfg.mask_invalid:
          assert core_caps[core] >= 0, f"Illegal core caps: {core_caps}"
        else:
          core_caps[core] = max(0, core_caps[core])
        del paired_qubits[qubit_set]
      
      while free_qubits:
        qubits = torch.tensor(free_qubits, dtype=torch.int, device=self.device).reshape((-1,1))
        qubits = torch.cat([qubits, -1*torch.ones_like(qubits)], dim=-1)
        _, _, log_pol = self.pred_model(
          qubits=qubits,
          prev_core_allocs=prev_core_allocs.expand((len(free_qubits), len(prev_core_allocs))),
          current_core_allocs=core_allocs.expand((len(free_qubits), len(prev_core_allocs))),
          core_capacities=core_caps.expand((len(free_qubits), hardware.n_cores)),
          core_connectivity=dev_core_con,
          circuit_emb=circ_embs[:,slice_idx,:,:],
        )
        qubit_set, core = self._sample_action(
          logits=log_pol,
          core_caps=core_caps,
          n_qubits=1,
          cfg=cfg
        )
        allocations[slice_idx,free_qubits[qubit_set]] = core
        core_allocs[free_qubits[qubit_set]] = core
        if ret_train_data:
          all_log_probs.append(log_pol[qubit_set, core])
        core_caps[core] -= 1
        if cfg.mask_invalid:
          assert core_caps[core] >= 0, f"Illegal core caps: {core_caps}"
        else:
          core_caps[core] = max(0, core_caps[core])
        del free_qubits[qubit_set]
    
    if ret_train_data:
      return torch.stack(all_log_probs)
  

  def optimize(
    self,
    circuit: Circuit,
    hardware: Hardware,
    cfg: DAConfig = DAConfig(),
  ) -> Tuple[torch.Tensor, float]:
    if circuit.n_qubits != hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {hardware.n_qubits} != {circuit.n_qubits}"
      ))
    self.pred_model.eval()
    circ_embs = circuit.embedding.to(self.device).unsqueeze(0)
    allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
    self._allocate(
      allocations=allocations,
      circ_embs=circ_embs,
      alloc_slices=circuit.alloc_slices,
      cfg=cfg,
      hardware=hardware,
      ret_train_data=False,
    )
    cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
    return allocations, cost


  def _update_best(
    self,
    val_cost: torch.Tensor,
    save_path:str,
    noise: float,
    it: int,
    optimizer: torch.optim.Optimizer
  ):
    vc_mean=val_cost.mean().item()
    chkpt_name = f"checkpt_{it}_{int(vc_mean*1000)}.pt"
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, chkpt_name))
    best_model = dict(
      val_cost=val_cost,
      vc_mean=vc_mean,
      model_state=deepcopy(self.pred_model.state_dict()),
      opt_state=deepcopy(optimizer.state_dict()),
      noise=noise,
    )
    print(f"saving as {chkpt_name}")
    return best_model


  def train(
    self,
    train_cfg: TrainConfig,
  ) -> dict[str, list]:
    self.iter_timer = Timer.get("_train_iter_timer")
    self.iter_timer.reset()
    optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=train_cfg.lr)
    opt_cfg = DAConfig(
      noise=train_cfg.initial_noise,
      mask_invalid=True,
      greedy=False,
    )
    data_log = dict(
      val_cost = [],
      loss = [],
      noise = [],
      t = []
    )
    init_t = time()
    best_model = dict(val_cost=None, vc_mean=None, model_state=None, opt_state=None, noise=None)
    save_path = self._make_save_dir(train_cfg.store_path, overwrite=False)

    try:
      for it in range(train_cfg.train_iters):
        # Train
        pheader = f"\033[2K\r[{it + 1}/{train_cfg.train_iters}]"
        self.iter_timer.start()
        cost_loss = self._train_batch(
          pheader=pheader,
          optimizer=optimizer,
          opt_cfg=opt_cfg,
          train_cfg=train_cfg,
        )

        # Validate
        if (it+1)%train_cfg.validate_each == 0:
          print(f"\033[2K\r      Running validation...", end='')
          with torch.no_grad():
            val_cost = self._validation(train_cfg=train_cfg)
          vc_mean = val_cost.mean().item()
          data_log['val_cost'].append(vc_mean)
          print(f"\033[2K\r      vc={vc_mean:.4f}, ", end='')
          if best_model['model_state'] is None:
            best_model = self._update_best(val_cost, save_path, opt_cfg.noise, it, optimizer)
          else:
            p = ttest_ind(val_cost.numpy(), best_model['val_cost'].numpy(), equal_var=False)[1]
            if p < 0.05:
              if vc_mean < best_model['vc_mean']:
                print(f"better than prev {best_model['vc_mean']:.4f} with p={p:.3f}, updating and ", end='')
                best_model = self._update_best(val_cost, save_path, opt_cfg.noise, it, optimizer)
              else:
                print(f"worse than prev {best_model['vc_mean']:.4f} with p={p:.3f}, backtracking")
                self.pred_model.load_state_dict(best_model['model_state'])
                optimizer.load_state_dict(best_model['opt_state'])
                opt_cfg.noise = best_model['noise']
            else:
              print(f"not enough significance p={p:.3f}, continuing")
          with open(os.path.join(save_path, "train_data.json"), "w") as f:
            json.dump(data_log, f, indent=2)

        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - it - 1)

        print((
          f"{pheader} l={cost_loss:.1f} \t n={opt_cfg.noise:.3f} t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left)"
        ))
        
        data_log['loss'].append(cost_loss)
        data_log['noise'].append(opt_cfg.noise)
        data_log['t'].append(time() - init_t)
        opt_cfg.noise *= train_cfg.noise_decrease_factor

    except KeyboardInterrupt as e:
      if 'y' not in input('\nGraceful shutdown? [y/n]: ').lower():
        raise e
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, "pred_mod.pt"))
    with open(os.path.join(save_path, "train_data.json"), "w") as f:
      json.dump(data_log, f, indent=2)
  

  def _train_batch(
    self,
    pheader: str,
    optimizer: torch.optim.Optimizer,
    opt_cfg: DAConfig,
    train_cfg: TrainConfig,
  ) -> float:
    self.pred_model.train()
    n_total = train_cfg.batch_size*train_cfg.group_size
    loss = 0

    for batch_i in range(train_cfg.batch_size):
      hardware = train_cfg.hardware_sampler.sample()
      train_cfg.circ_sampler.num_lq = hardware.n_qubits
      circuit = train_cfg.circ_sampler.sample()
      circ_embs = circuit.embedding.to(self.device).unsqueeze(0)
      all_costs = torch.empty([train_cfg.group_size], device=self.device)
      action_log_probs = torch.empty([train_cfg.group_size], device=self.device)

      for group_i in range(train_cfg.group_size):
        opt_n = group_i + batch_i*train_cfg.group_size
        print(f"{pheader} ns={circuit.n_slices} nq={hardware.n_qubits} nc={hardware.n_cores} Optimizing {opt_n + 1}/{n_total}", end='')
        allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
        log_probs = self._allocate(
          allocations=allocations,
          circ_embs=circ_embs,
          alloc_slices=circuit.alloc_slices,
          cfg=opt_cfg,
          hardware=hardware,
          ret_train_data=True,
        )
        cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
        all_costs[group_i] = cost/(circuit.n_gates_norm + 1)
        action_log_probs[group_i] = torch.sum(log_probs)

      all_costs = (all_costs - all_costs.mean()) / (all_costs.std(unbiased=True) + 1e-8)
      loss += torch.sum(all_costs*action_log_probs)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_norm=1)
    optimizer.step()
    return loss.item()
  

  def _validation(self, train_cfg: TrainConfig) -> float:
    da_cfg = DAConfig()
    norm_costs = torch.empty([len(train_cfg.validation_circuits)])
    for i, circ in enumerate(train_cfg.validation_circuits):
      norm_costs[i] = self.optimize(circ, cfg=da_cfg, hardware=train_cfg.validation_hardware)[1]/(circ.n_gates_norm + 1)
    return norm_costs