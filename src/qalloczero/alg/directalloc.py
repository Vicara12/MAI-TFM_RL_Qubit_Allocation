import os
import json
import torch
import gc
import warnings
from enum import Enum
from time import time
from typing import Self, Tuple, Optional, Any
from copy import deepcopy
from dataclasses import dataclass, asdict
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost, get_all_checkpoints
from scipy.stats import ttest_ind
from utils.timer import Timer
from sampler.hardwaresampler import HardwareSampler
from sampler.circuitsampler import CircuitSampler
from qalloczero.alg.ts import ModelConfigs
from qalloczero.models.predmodel import PredictionModel



from utils.memory import all_top, print_ram_usage


@dataclass
class DAConfig:
  noise: float = 0.0
  mask_invalid: bool = True
  greedy: bool = True


class DirectAllocator:

  class Mode(Enum):
    Sequential = 0
    Parallel   = 1

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
    min_noise: float
    circ_sampler: CircuitSampler
    lr: float
    inv_mov_penalization: float
    hardware_sampler: HardwareSampler
    mask_invalid: bool
    dropout: float = 0.0


  def __init__(
    self,
    device: str = "cpu",
    model_cfg: ModelConfigs = ModelConfigs(),
    mode: Mode = Mode.Sequential,
  ):
    self.model_cfg = model_cfg
    self.pred_model = PredictionModel(
      embed_size=model_cfg.embed_size,
      num_heads=model_cfg.num_heads,
      num_layers=model_cfg.num_layers,
    )
    self.pred_model.to(device)
    self.mode = mode
  

  @property
  def device(self) -> torch.device:
    return next(self.pred_model.parameters()).device


  def _save_model_cfg(self, path: str):
    params = dict(
      embed_size=self.model_cfg.embed_size,
      num_heads=self.model_cfg.num_heads,
      num_layers=self.model_cfg.num_layers,
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
  def load(path: str, device: str = "cuda", checkpoint: Optional[int] = None) -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "optimizer_conf.json"), "r") as f:
      params = json.load(f)
    model_cfg = ModelConfigs(
      embed_size=params['embed_size'],
      num_heads=params['num_heads'],
      num_layers=params['num_layers'],
    )
    loaded = DirectAllocator(device=device, model_cfg=model_cfg)
    model_file = "pred_mod.pt"
    if checkpoint is not None:
      chpt_files = get_all_checkpoints(path)
      if checkpoint == -1:
        checkpoint = max(list(chpt_files.keys()))
      elif checkpoint not in chpt_files.keys():
        raise Exception(f'Checkpoint {checkpoint} not found: {", ".join(list(chpt_files.keys()))}')
      model_file = chpt_files[checkpoint]
    loaded.pred_model.load_state_dict(
      torch.load(
        os.path.join(path, model_file),
        weights_only=False,
        map_location=device,
      )
    )
    return loaded
  

  def set_mode(self, mode: Mode):
    self.mode = mode
    return self


  def _sample_action_sequential(
    self,
    pol: torch.Tensor,
    core_caps: torch.Tensor,
    n_qubits: int,
    cfg: DAConfig,
  ) -> Tuple[int, torch.Tensor, torch.Tensor]:
    # Set prior of cores that do not have space for this alloc to zero
    valid_cores = (core_caps >= n_qubits)
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
      pass
    else:
      pol /= sum_pol
    core = pol.argmax().item() if cfg.greedy else torch.distributions.Categorical(pol).sample()
    valid = valid_cores[core]
    return core, pol, valid


  def _sample_action_parallel(
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
    valid = valid_cores[action]
    (qubit_set, core) = torch.unravel_index(action, logits.shape)
    return qubit_set.item(), core.item(), valid


  def _allocate_sequential(
    self,
    allocations: torch.Tensor,
    circ_embs: torch.Tensor,
    next_interactions: torch.Tensor,
    alloc_steps: torch.Tensor,
    cfg: DAConfig,
    hardware: Hardware,
    ret_train_data: bool,
    verbose: bool = False,
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
      all_probs = []
      all_valid = []
    prev_slice = -1
    for step, (slice_idx, qubit0, qubit1, _) in enumerate(alloc_steps):

      if verbose:
        print((f"\033[2K\r - Optimization step {step+1}/{len(alloc_steps)} "
               f"({int(100*(step+1)/len(alloc_steps))}%)"), end="")
        
      if prev_slice != slice_idx:
        prev_core_allocs = core_allocs
        core_allocs = hardware.n_cores * torch.ones_like(core_allocs)
        core_caps = core_caps_orig.clone()
      pol, _, log_pol = self.pred_model(
        qubits=torch.tensor([qubit0, qubit1], dtype=torch.int, device=self.device).unsqueeze(0),
        prev_core_allocs=prev_core_allocs.unsqueeze(0),
        current_core_allocs=core_allocs.unsqueeze(0),
        core_capacities=core_caps.unsqueeze(0),
        core_connectivity=hardware.core_connectivity.to(self.device),
        circuit_emb=circ_embs[:,slice_idx],
        next_interactions=next_interactions[:,slice_idx]
      )
      pol = pol.squeeze(0)
      log_pol = log_pol.squeeze(0)
      n_qubits = (1 if qubit1 == -1 else 2)
      action, pol, valid = self._sample_action_sequential(
        pol=pol,
        core_caps=core_caps,
        n_qubits=n_qubits,
        cfg=cfg
      )
      allocations[slice_idx,qubit0] = action
      core_allocs[qubit0] = action
      if qubit1 != -1:
        allocations[slice_idx,qubit1] = action
        core_allocs[qubit1] = action
      if ret_train_data:
        all_probs.append(log_pol[action])
        all_valid.append(valid)
      if cfg.mask_invalid:
        core_caps[action] = core_caps[action] - n_qubits
        assert core_caps[action] >= 0, f"Illegal core caps: {core_caps}"
      else:
        core_caps[action] = max(0, core_caps[action] - n_qubits)
      prev_slice = slice_idx
    if verbose:
      print('\033[2K\r', end='')
    if ret_train_data:
      return torch.stack(all_probs), torch.tensor(all_valid)


  def _allocate_parallel(
    self,
    allocations: torch.Tensor,
    circ_embs: torch.Tensor,
    next_interactions: torch.Tensor,
    alloc_slices: list[tuple[int, list[int], list[tuple[int,int]]]],
    cfg: DAConfig,
    hardware: Hardware,
    ret_train_data: bool,
    verbose: bool = False,
  ) -> Optional[torch.Tensor]:
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
      all_valid = []
      
    dev_core_con = hardware.core_connectivity.to(self.device)
    step = 0
    n_steps = sum(len(s[1])+len(s[2]) for s in alloc_slices)
    
    for slice_idx, (_, free_qubits, paired_qubits) in enumerate(alloc_slices):
      prev_core_allocs = core_allocs
      core_allocs = hardware.n_cores * torch.ones_like(core_allocs)
      core_caps = core_caps_orig.clone()
      paired_qubits = list(paired_qubits)
      free_qubits = list(free_qubits)
      
      while paired_qubits:
        if verbose:
          print((f"\033[2K\r - Optimization step {step+1}/{n_steps} ({int(100*(step+1)/n_steps)}%)"), end="")
          step += 1
        pol, _, log_pol = self.pred_model(
          qubits=torch.tensor(paired_qubits, dtype=torch.int, device=self.device),
          prev_core_allocs=prev_core_allocs.expand((len(paired_qubits), len(prev_core_allocs))),
          current_core_allocs=core_allocs.expand((len(paired_qubits), len(prev_core_allocs))),
          core_capacities=core_caps.expand((len(paired_qubits), hardware.n_cores)),
          core_connectivity=dev_core_con,
          circuit_emb=circ_embs[:,slice_idx,:,:].expand((len(paired_qubits), -1, -1)),
          next_interactions=next_interactions[:,slice_idx,:,:].expand((len(paired_qubits), -1, -1)),
        )
        qubit_set, core, valid = self._sample_action_parallel(
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
          all_valid.append(valid)
        core_caps[core] -= 2
        if cfg.mask_invalid:
          assert core_caps[core] >= 0, f"Illegal core caps: {core_caps}"
        else:
          core_caps[core] = max(0, core_caps[core])
        del paired_qubits[qubit_set]
      
      while free_qubits:
        if verbose:
          print((f"\033[2K\r - Optimization step {step+1}/{n_steps} ({int(100*(step+1)/n_steps)}%)"), end="")
          step += 1

        qubits = torch.tensor(free_qubits, dtype=torch.int, device=self.device).reshape((-1,1))
        qubits = torch.cat([qubits, -1*torch.ones_like(qubits)], dim=-1)
        _, _, log_pol = self.pred_model(
          qubits=qubits,
          prev_core_allocs=prev_core_allocs.expand((len(free_qubits), len(prev_core_allocs))),
          current_core_allocs=core_allocs.expand((len(free_qubits), len(prev_core_allocs))),
          core_capacities=core_caps.expand((len(free_qubits), hardware.n_cores)),
          core_connectivity=dev_core_con,
          circuit_emb=circ_embs[:,slice_idx,:,:],
          next_interactions=next_interactions[:,slice_idx,:,:],
        )
        qubit_set, core, valid = self._sample_action_parallel(
          logits=log_pol,
          core_caps=core_caps,
          n_qubits=1,
          cfg=cfg
        )
        allocations[slice_idx,free_qubits[qubit_set]] = core
        core_allocs[free_qubits[qubit_set]] = core
        if ret_train_data:
          all_log_probs.append(log_pol[qubit_set, core])
          all_valid.append(valid)
        core_caps[core] -= 1
        if cfg.mask_invalid:
          assert core_caps[core] >= 0, f"Illegal core caps: {core_caps}"
        else:
          core_caps[core] = max(0, core_caps[core])
        del free_qubits[qubit_set]
    if verbose:
      print('\033[2K\r', end='')
    if ret_train_data:
      return torch.stack(all_log_probs), torch.tensor(all_valid)
  

  def _allocate(
    self,
    allocations: torch.Tensor,
    circuit: Circuit,
    cfg: DAConfig,
    hardware: Hardware,
    ret_train_data: bool,
    verbose: bool = False
  ):
    if self.mode == DirectAllocator.Mode.Sequential:
      return self._allocate_sequential(
        allocations=allocations,
        circ_embs=circuit.embedding.to(self.device).unsqueeze(0),
        next_interactions=circuit.next_interaction.to(self.device).unsqueeze(0),
        alloc_steps=circuit.alloc_steps,
        cfg=cfg,
        hardware=hardware,
        ret_train_data=ret_train_data,
        verbose=verbose,
      )
    else:
      return self._allocate_parallel(
        allocations=allocations,
        circ_embs=circuit.embedding.to(self.device).unsqueeze(0),
        next_interactions=circuit.next_interaction.to(self.device).unsqueeze(0),
        alloc_slices=circuit.alloc_slices,
        cfg=cfg,
        hardware=hardware,
        ret_train_data=ret_train_data,
        verbose=verbose,
      )


  def optimize(
    self,
    circuit: Circuit,
    hardware: Hardware,
    cfg: DAConfig = DAConfig(),
    verbose: bool = False
  ) -> Tuple[torch.Tensor, float]:
    if circuit.n_qubits != hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {hardware.n_qubits} != {circuit.n_qubits}"
      ))
    self.pred_model.eval()
    allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
    self._allocate(
      allocations=allocations,
      circuit=circuit,
      cfg=cfg,
      hardware=hardware,
      ret_train_data=False,
      verbose=verbose,
    )
    cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
    return allocations, cost


  def _update_best(
    self,
    val_cost: torch.Tensor,
    save_path:str,
    it: int,
  ):
    vc_mean=val_cost.mean().item()
    chkpt_name = f"checkpt_{it+1}_{int(vc_mean*1000)}.pt"
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, chkpt_name))
    best_model = dict(
      val_cost=val_cost,
      vc_mean=vc_mean,
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
      mask_invalid=train_cfg.mask_invalid,
      greedy=False,
    )
    data_log = dict(
      train_cfg = dict(
        inference_mode=str(self.mode),
        train_iters=train_cfg.train_iters,
        batch_size=train_cfg.batch_size,
        group_size=train_cfg.group_size,
        validate_each=train_cfg.validate_each,
        initial_noise=train_cfg.initial_noise,
        noise_decrease_factor=train_cfg.noise_decrease_factor,
        lr=train_cfg.lr,
        inv_mov_penalization=train_cfg.inv_mov_penalization,
        hws_nqubits=train_cfg.hardware_sampler.max_nqubits,
        hws_range_ncores=train_cfg.hardware_sampler.range_ncores,
        min_noise=train_cfg.min_noise,
        mask_invalid=train_cfg.mask_invalid,
        dropout=train_cfg.dropout,
      ),
      val_cost = [],
      loss = [],
      cost_loss = [],
      val_loss = [],
      noise = [],
      vm=[],
      t = []
    )
    self.pred_model.set_dropout(train_cfg.dropout)
    init_t = time()
    best_model = dict(val_cost=None, vc_mean=None)
    save_path = self._make_save_dir(train_cfg.store_path, overwrite=False)

    try:
      for it in range(train_cfg.train_iters):

        print_ram_usage()

        # Train
        pheader = f"\033[2K\r[{it + 1}/{train_cfg.train_iters}]"
        self.iter_timer.start()

        loss, cost_loss, val_loss, vm_ratio = self._train_batch(
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
          if best_model['val_cost'] is None:
            best_model = self._update_best(val_cost, save_path, it)
          else:
            p = ttest_ind(val_cost.numpy(), best_model['val_cost'].numpy(), equal_var=False)[1]
            if p < 0.2:
              if vc_mean < best_model['vc_mean']:
                print(f"better than prev {best_model['vc_mean']:.4f} with p={p:.3f}, updating and ", end='')
                best_model = self._update_best(val_cost, save_path, it)
              else:
                print(f"worse than prev {best_model['vc_mean']:.4f} with p={p:.3f}, ", end='')
                self._update_best(val_cost, save_path, it)
            else:
              print(f"not enough significance wrt prev={best_model['vc_mean']:.4f} p={p:.3f}, ", end='')
              self._update_best(val_cost, save_path, it)
          with open(os.path.join(save_path, "train_data.json"), "w") as f:
            json.dump(data_log, f, indent=2)

        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - it - 1)

        print((
          f"{pheader} l={loss:.1f} (c={cost_loss:.1f} v={val_loss:.1f}) \t n={opt_cfg.noise:.3f} "
          f"vm={vm_ratio:.2f} t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left)"
        ))
        
        data_log['loss'].append(loss)
        data_log['cost_loss'].append(cost_loss)
        data_log['val_loss'].append(val_loss)
        data_log['noise'].append(opt_cfg.noise)
        data_log['t'].append(time() - init_t)
        data_log['vm'].append(vm_ratio)
        opt_cfg.noise = max(train_cfg.min_noise, opt_cfg.noise*train_cfg.noise_decrease_factor)

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
    inv_pen = train_cfg.inv_mov_penalization

    while True:
      total_loss = 0
      total_cost_loss = 0
      total_valid_loss = 0
      valid_moves_ratio = 0
      optimizer.zero_grad()
      try:
        # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        for batch_i in range(train_cfg.batch_size):
          hardware = train_cfg.hardware_sampler.sample()
          train_cfg.circ_sampler.num_lq = hardware.n_qubits
          circuit = train_cfg.circ_sampler.sample()
          all_costs = torch.empty([train_cfg.group_size], device=self.device)
          action_log_probs = torch.empty([train_cfg.group_size], device=self.device)
          inv_moves_sum = torch.empty([train_cfg.group_size], device=self.device)

          for group_i in range(train_cfg.group_size):
            opt_n = group_i + batch_i*train_cfg.group_size
            print(f"{pheader} ns={circuit.n_slices} nq={hardware.n_qubits} nc={hardware.n_cores} Optimizing {opt_n + 1}/{n_total}", end='')
            allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
            log_probs, valid_moves = self._allocate(
              allocations=allocations,
              circuit=circuit,
              cfg=opt_cfg,
              hardware=hardware,
              ret_train_data=True,
            )
            cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
            all_costs[group_i] = cost/(circuit.n_gates_norm + 1)
            action_log_probs[group_i] = torch.sum(log_probs[valid_moves])
            inv_moves_sum[group_i] = torch.sum(log_probs[~valid_moves])
            valid_moves_ratio += valid_moves.float().mean().item()

          all_costs = (all_costs - all_costs.mean()) / (all_costs.std(unbiased=True) + 1e-8)
          print(f'\ncosts: {" ".join([f"{v:.3f}" for v in all_costs.tolist()])}')
          cost_loss = torch.sum(all_costs*action_log_probs)
          total_cost_loss += cost_loss.item()
          valid_loss = torch.sum(inv_moves_sum)
          total_valid_loss += valid_loss.item()
          loss = ((1 - inv_pen)*cost_loss + inv_pen*valid_loss)/train_cfg.batch_size
          loss.backward()
          total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_norm=1)
        optimizer.step()
        break
      except torch.cuda.OutOfMemoryError:
        print(" Ran out of VRAM! Retrying...")
        if 'loss' in locals(): del loss
        if 'cost_loss' in locals(): del cost_loss
        if 'valid_loss' in locals(): del valid_loss
        if 'action_log_probs' in locals(): del action_log_probs
        if 'inv_moves_sum' in locals(): del inv_moves_sum
        if 'log_probs' in locals(): del log_probs
        if 'allocations' in locals(): del allocations
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return total_loss, total_cost_loss, total_valid_loss, valid_moves_ratio/(train_cfg.batch_size*train_cfg.group_size)
  

  def _validation(self, train_cfg: TrainConfig) -> float:
    da_cfg = DAConfig()
    norm_costs = torch.empty([len(train_cfg.validation_circuits)])
    for i, circ in enumerate(train_cfg.validation_circuits):
      norm_costs[i] = self.optimize(circ, cfg=da_cfg, hardware=train_cfg.validation_hardware)[1]/(circ.n_gates_norm + 1)
    return norm_costs