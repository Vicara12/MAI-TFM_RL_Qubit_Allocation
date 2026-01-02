import os
import json
import torch
import torch.multiprocessing as tmp
import warnings
import random
import copy
from math import ceil
from enum import Enum
from time import time, sleep
from typing import Self, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost, get_all_checkpoints
from scipy.stats import ttest_ind
from utils.timer import Timer
from utils.memory import get_ram_usage
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

  class Mode(Enum):
    Sequential = 0
    Parallel   = 1

  @dataclass
  class TrainConfig:
    train_iters: int
    group_size: int
    n_workers: int
    worker_devices: list[str]
    train_device: str
    ett: int # Take n circuits from the extremes of group to train
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
  

  class TrainDataGatherer:
    def __init__(self, data_queue: tmp.Queue, processes: list[tmp.Process], n_items: int):
      self.data_queue = data_queue
      self.processes = processes
      self.n_items = n_items
    
    def abort(self):
      for proc in self.processes:
        proc.terminate()
      while not self.data_queue.empty():
        self.data_queue.get()
    
    def __del__(self):
      self.abort()

    @property
    def finished(self) -> bool:
      return self.data_queue.qsize() >= self.n_items
    
    @property
    def healty(self) -> bool:
      return all(map(lambda p: p.is_alive(), self.processes))

    @property
    def data(self) -> list:
      if not self.healty:
        raise Exception("some worker has failed")
      if not self.finished:
        raise Exception("gatherer has not finished")
      items = [None]*self.n_items
      while not self.data_queue.empty():
        dq_item = self.data_queue.get()
        idx = dq_item['opt_n']
        if idx >= self.n_items:
          raise Exception(f"found value outside gatherer range: {idx} >= {self.n_items}")
        del dq_item['opt_n']
        items[idx] = dq_item
      self.abort()
      if any(map(lambda x: x is None, items)):
        raise Exception("missing some value in gatherer result list")
      return items

    


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
    old_path = path
    if os.path.isdir(path):
      if not overwrite:
        i = 2
        while os.path.isdir(path + f"_v{i}"):
          i += 1
        path += f"_v{i}"
        os.makedirs(path)
        warnings.warn(f"Provided folder \"{old_path}\" already exists, saving as \"{path}\"")
      else:
        warnings.warn(f"Provided folder \"{old_path}\" already exists, overwriting previous save file")
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
  ) -> Optional[dict[str, torch.Tensor]]:
    core_caps_orig = hardware.core_capacities.to(self.device)
    core_allocs = torch.zeros(
      [hardware.n_cores, hardware.n_qubits],
      dtype=torch.float,
      device=self.device,
    )
    prev_core_allocs = None
    core_caps = None
    if ret_train_data:
      all_qubits = []
      all_actions = []
      all_slices = []
      all_log_probs = []
      all_valid = []
    prev_slice = -1
    for step, (slice_idx, qubit0, qubit1, _) in enumerate(alloc_steps):

      if verbose:
        print((f"\033[2K\r - Optimization step {step+1}/{len(alloc_steps)} "
               f"({int(100*(step+1)/len(alloc_steps))}%)"), end="")
        
      if prev_slice != slice_idx:
        prev_core_allocs = core_allocs
        core_allocs = torch.zeros_like(core_allocs)
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
      core_allocs[action, qubit0] = 1
      if qubit1 != -1:
        allocations[slice_idx,qubit1] = action
        core_allocs[action, qubit1] = 1
      if ret_train_data:
        all_qubits.append((qubit0, qubit1))
        all_actions.append(action)
        all_slices.append(slice_idx)
        all_log_probs.append(log_pol[action].item())
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
      return dict(
        all_qubits = torch.tensor(all_qubits).cpu(),
        all_actions = torch.tensor(all_actions).cpu(),
        all_slices = torch.tensor(all_slices).cpu(),
        all_log_probs = torch.tensor(all_log_probs).cpu(),
        all_valid = torch.tensor(all_valid).cpu(),
      )


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
  ) -> Optional[dict[str, torch.Tensor]]:
    core_caps_orig = hardware.core_capacities.to(self.device)
    core_allocs = torch.zeros(
      [hardware.n_cores, hardware.n_qubits],
      dtype=torch.float,
      device=self.device,
    )
    prev_core_allocs = None
    core_caps = None
    if ret_train_data:
      all_qubits = []
      all_actions = []
      all_slices = []
      all_log_probs = []
      all_valid = []
      
    dev_core_con = hardware.core_connectivity.to(self.device)
    step = 0
    n_steps = sum(len(s[1])+len(s[2]) for s in alloc_slices)
    
    for slice_idx, (_, free_qubits, paired_qubits) in enumerate(alloc_slices):
      prev_core_allocs = core_allocs
      core_allocs = torch.zeros_like(core_allocs)
      core_caps = core_caps_orig.clone()
      paired_qubits = list(paired_qubits)
      free_qubits = list(free_qubits)
      
      while paired_qubits:
        if verbose:
          print((f"\033[2K\r - Optimization step {step+1}/{n_steps} ({int(100*(step+1)/n_steps)}%)"), end="")
          step += 1
        pol, _, log_pol = self.pred_model(
          qubits=torch.tensor(paired_qubits, dtype=torch.int, device=self.device),
          prev_core_allocs=prev_core_allocs.expand((len(paired_qubits), -1, -1)),
          current_core_allocs=core_allocs.expand((len(paired_qubits), -1, -1)),
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
        core_allocs[core, paired_qubits[qubit_set][0]] = 1
        allocations[slice_idx,paired_qubits[qubit_set][1]] = core
        core_allocs[core, paired_qubits[qubit_set][1]] = 1
        if ret_train_data:
          all_qubits.append((paired_qubits[qubit_set][0], paired_qubits[qubit_set][1]))
          all_actions.append(core)
          all_slices.append(slice_idx)
          all_log_probs.append(log_pol[qubit_set, core].item())
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
          prev_core_allocs=prev_core_allocs.expand((len(free_qubits), -1, -1)),
          current_core_allocs=core_allocs.expand((len(free_qubits), -1, -1)),
          core_capacities=core_caps.expand((len(free_qubits), hardware.n_cores)),
          core_connectivity=dev_core_con,
          circuit_emb=circ_embs[:,slice_idx,:,:].expand((len(free_qubits), -1, -1)),
          next_interactions=next_interactions[:,slice_idx,:,:],
        )
        qubit_set, core, valid = self._sample_action_parallel(
          logits=log_pol,
          core_caps=core_caps,
          n_qubits=1,
          cfg=cfg
        )
        allocations[slice_idx,free_qubits[qubit_set]] = core
        core_allocs[core, free_qubits[qubit_set]] = 1
        if ret_train_data:
          all_qubits.append((free_qubits[qubit_set], -1))
          all_actions.append(core)
          all_slices.append(slice_idx)
          all_log_probs.append(log_pol[qubit_set, core].item())
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
      return dict(
        all_qubits = torch.tensor(all_qubits).cpu(),
        all_actions = torch.tensor(all_actions).cpu(),
        all_slices = torch.tensor(all_slices).cpu(),
        all_log_probs = torch.tensor(all_log_probs).cpu(),
        all_valid = torch.tensor(all_valid).cpu(),
      )


  def _allocate(
    self,
    allocations: torch.Tensor,
    circuit: Circuit,
    cfg: DAConfig,
    hardware: Hardware,
    ret_train_data: bool,
    verbose: bool = False
  ) -> Optional[dict[str, torch.Tensor]]:
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
  

  def optimize_mult(
    self,
    circuits: list[Circuit],
    hardware: Hardware,
    n_workers: int,
    devices: list[str],
    cfg: DAConfig = DAConfig(),
  ):
    data_future = self._launch_opt_workers(
      opt_cfg=cfg,
      n_workers=n_workers,
      opt_devices=devices,
      circuits=circuits,
      hardware=hardware,
      ret_train_data=False,
    )
    while not data_future.finished:
      sleep(0.1)
    return [(item['allocations'], item['cost']) for item in data_future.data]


  def _train_cfg_to_dict(self, train_cfg: TrainConfig) -> dict:
    d = {}
    for k,v in asdict(train_cfg).items():
      try:
        d[k] = float(v)
        continue
      except:
        pass
      try:
        d[k] = str(v)
        continue
      except:
        pass
    return d


  def train(
    self,
    train_cfg: TrainConfig,
  ) -> dict[str, list]:
    self.iter_timer = Timer.get("_train_iter_timer")
    self.iter_timer.reset()
    tmp.set_start_method('spawn', force=True)
    # For some reason I need to send the model through CPU first otherwise parameters get filled with 0
    self.pred_model.cpu()
    self.pred_model.to(train_cfg.train_device)
    optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=train_cfg.lr)
    opt_cfg = DAConfig(
      noise=train_cfg.initial_noise,
      mask_invalid=train_cfg.mask_invalid,
      greedy=False,
    )
    data_log = dict(
      train_cfg = self._train_cfg_to_dict(train_cfg),
      val_cost = [],
      loss = [],
      cost_loss = [],
      val_loss = [],
      noise = [],
      vm=[],
      t = []
    )
    assert (train_cfg.group_size%train_cfg.n_workers == 0), f"n_workers must divide group_size"
    self.pred_model.set_dropout(train_cfg.dropout)
    init_t = time()
    best_model = dict(val_cost=None, vc_mean=None)
    save_path = self._make_save_dir(train_cfg.store_path, overwrite=False)
    train_data = None

    try:
      for it in range(train_cfg.train_iters):
        # Train
        pheader = f"\033[2K\r[{it + 1}/{train_cfg.train_iters}]"
        self.iter_timer.start()

        loss, cost_loss, val_loss, vm_ratio, train_data = self._train_batch(
          pheader=pheader,
          optimizer=optimizer,
          opt_cfg=opt_cfg,
          train_cfg=train_cfg,
          train_data=train_data,
        )

        # Validate
        if (it+1)%train_cfg.validate_each == 0:
          best_model = self._validation(pheader, data_log, save_path, train_cfg, it, best_model)

        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - it - 1)

        print((
          f"{pheader} l={loss:.4f} (c={cost_loss:.4f} v={val_loss:.4f}) \t n={opt_cfg.noise:.3f} "
          f"vm={vm_ratio:.3f} t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} left) "
          f"ram={get_ram_usage():.2f}GB"
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
    self.pred_model
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, "pred_mod.pt"))
    with open(os.path.join(save_path, "train_data.json"), "w") as f:
      json.dump(data_log, f, indent=2)
  

  def _opt_parallel_worker(
    self,
    n: int,
    device: str,
    circuits: list[Circuit],
    cfg: DAConfig,
    hardware: Hardware,
    data_queue: tmp.Queue,
    ret_train_data: bool,
  ):
    self.pred_model.cpu()
    self.pred_model.to(device)
    with torch.no_grad():
      for i, circuit in enumerate(circuits):
        allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
        train_data = self._allocate(
          allocations=allocations,
          circuit=circuit,
          cfg=cfg,
          hardware=hardware,
          ret_train_data=ret_train_data,
          verbose=False,
        )
        cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
        opt_metadata = {'opt_n': n + i, 'cost': cost, 'allocations': allocations}
        data_queue.put(opt_metadata | (train_data if train_data is not None else {}))
    # Wait until parent reads data and sends terminate signal
    while True:
      sleep(10)
  

  def _launch_opt_workers(
    self,
    opt_cfg: DAConfig,
    n_workers: int,
    opt_devices: list[str],
    circuits: list[Circuit],
    hardware: Hardware,
    ret_train_data: bool,
  ) -> TrainDataGatherer:
    data_queue = tmp.Queue()
    workers = []
    n_per_worker = ceil(len(circuits)/n_workers)
    try:
      for i in range(n_workers):
        workers.append(tmp.Process(
          target=self._opt_parallel_worker,
          args=(
            i*n_per_worker,
            opt_devices[i%len(opt_devices)],
            circuits[i*n_per_worker:(i + 1)*n_per_worker],
            opt_cfg,
            hardware,
            data_queue,
            ret_train_data,
          ),
        ))
        workers[-1].start()
      gatherer_obj = DirectAllocator.TrainDataGatherer(data_queue, workers, len(circuits))
    except:
      for worker in workers:
        worker.terminate()
    return gatherer_obj


  def _rebuild_core_info_worker(
    self,
    hardware: Hardware,
    data: list[dict],
    data_queue: tmp.Queue
  ):
    for k,v in enumerate(data):
      core_info = {
        'sample': k,
        'prev_core_allocs': [],
        'current_core_allocs': [],
        'core_capacities': [],
      }
      current_core_allocs = torch.zeros([hardware.n_cores, hardware.n_qubits])
      prev_core_allocs = None
      core_capacities = None
      prev_slice_i = -1

      for ((q0,q1), core, slice_i) in zip(v['all_qubits'], v['all_actions'], v['all_slices']):
        if slice_i != prev_slice_i:
          prev_core_allocs = current_core_allocs
          current_core_allocs = torch.zeros_like(current_core_allocs)
          core_capacities = hardware.core_capacities.clone()
          prev_slice_i = slice_i
        core_info['prev_core_allocs'].append(prev_core_allocs)
        core_info['current_core_allocs'].append(current_core_allocs)
        core_info['core_capacities'].append(core_capacities)
        current_core_allocs = current_core_allocs.clone()
        core_capacities = core_capacities.clone()
        current_core_allocs[core, q0] = 1
        core_capacities[core] -= 1
        if q1 != -1:
          current_core_allocs[core, q1] = 1
          core_capacities[core] -= 1
        core_capacities[core] = max(core_capacities[core], 0)
      core_info['prev_core_allocs'] = torch.stack(core_info['prev_core_allocs'])
      core_info['current_core_allocs'] = torch.stack(core_info['current_core_allocs'])
      core_info['core_capacities'] = torch.stack(core_info['core_capacities'])
      data_queue.put(core_info)
    # Wait until parent finishes and calls terminate on process
    while True:
      sleep(10)

  def _rebuild_core_info(
    self,
    hardware: Hardware,
    data: list[dict]
  ) -> tuple[tmp.Process, tmp.Queue]:
    data_queue = tmp.Queue()
    p = tmp.Process(target=self._rebuild_core_info_worker, args=(hardware, data, data_queue))
    p.start()
    return p, data_queue

  def _train_batch(
    self,
    pheader: str,
    optimizer: torch.optim.Optimizer,
    opt_cfg: DAConfig,
    train_cfg: TrainConfig,
    train_data: Optional[tuple[Hardware,Circuit,list[dict]]],
  ) -> float:
    # Sample hardware and circuit and optimize groups size times
    next_hardware = train_cfg.hardware_sampler.sample()
    train_cfg.circ_sampler.num_lq = next_hardware.n_qubits
    next_circuit = train_cfg.circ_sampler.sample()
    # Execute workers with a clone of pred model so that we can update the main one
    self.pred_model, p_model_backup = copy.deepcopy(self.pred_model), self.pred_model
    self.pred_model.eval()
    print(f"{pheader} ns={next_circuit.n_slices} nq={next_hardware.n_qubits} nc={next_hardware.n_cores} Optimizing {train_cfg.group_size} circuits...", end='')
    next_train_data_future = self._launch_opt_workers(
      opt_cfg=opt_cfg,
      n_workers=train_cfg.n_workers,
      opt_devices=train_cfg.worker_devices,
      circuits=[next_circuit]*train_cfg.group_size,
      hardware=next_hardware,
      ret_train_data=True,
    )
    self.pred_model = p_model_backup

    total_cost_loss = 0
    total_vm_loss = 0
    total_loss = 0
    vm_ratio = 0
    if train_data is not None:
      hardware, circuit, batch_data = train_data
      # Normalize costs by number of gates
      for i in range(len(batch_data)):
        batch_data[i]['cost'] /= (circuit.n_gates_norm + 1)

      # Normalize cost vector and select 16 best and worst samples
      overcost = torch.tensor([v['cost'] for v in batch_data])
      overcost = (overcost - overcost.mean().item()) / max(overcost.std(unbiased=True), 1e-5)
      # Clamp more upwards because it's very easy to mess cost up badly but difficult to improve it
      overcost = torch.clamp(overcost, -5.0, 2.0)
      batch_data = [(item | {"overcost": ov.item()}) for item, ov in zip(batch_data, overcost)]
      batch_data = sorted(batch_data, key=lambda x: x['overcost'])
      ett = train_cfg.ett
      # Take only ett//2 top and worst samples from group to train
      batch_data = batch_data if len(batch_data) < ett else batch_data[:ett//2] + batch_data[-ett//2:]

      self.pred_model.train()
      inv_pen = train_cfg.inv_mov_penalization
      optimizer.zero_grad()

      core_con = hardware.core_connectivity.to(train_cfg.train_device)
      slices = batch_data[0]['all_slices']
      ce = circuit.embedding[slices].to(train_cfg.train_device)
      ni = circuit.next_interaction[slices].to(train_cfg.train_device)
      p, data_queue = self._rebuild_core_info(hardware, batch_data)
      try:
        n_actions = sum(len(di['all_actions']) for di in batch_data)
        for _ in range(len(batch_data)):
          while data_queue.empty():
            if not p.is_alive():
              raise Exception('core info rebuilt process has died')
            sleep(0.1)
          core_info = data_queue.get()
          data_i = batch_data[core_info['sample']]
          _, _, log_probs = self.pred_model(
            qubits=data_i['all_qubits'].to(train_cfg.train_device),
            prev_core_allocs=core_info['prev_core_allocs'].to(train_cfg.train_device),
            current_core_allocs=core_info['current_core_allocs'].to(train_cfg.train_device),
            core_capacities=core_info['core_capacities'].to(train_cfg.train_device),
            core_connectivity=core_con,
            circuit_emb=ce,
            next_interactions=ni,
          )
          actions = data_i['all_actions'].to(train_cfg.train_device)
          log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
          ratios = torch.clamp(
            torch.exp(log_probs - data_i['all_log_probs'].to(train_cfg.train_device)),
            min=0,
            max=5
          )
          valid_mvs = data_i['all_valid'].to(train_cfg.train_device)

          cost_loss = torch.sum(torch.max(
            data_i['overcost'] * ratios[valid_mvs],
            data_i['overcost'] * torch.clamp(ratios[valid_mvs], 0.8, 1.2)
          )) / n_actions
          vm_loss = torch.sum(torch.clamp(ratios[~valid_mvs], min=None, max=1.2)) / n_actions
          loss = ((1 - inv_pen) * cost_loss + inv_pen * vm_loss)
          loss.backward()
          total_cost_loss += cost_loss.item()
          total_vm_loss += vm_loss.item()
          total_loss += loss.item()
          vm_ratio += valid_mvs.float().sum().item()/n_actions
      except:
        next_train_data_future.abort()
      finally:
        p.terminate()
      torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_norm=1)
      optimizer.step()

    while not next_train_data_future.finished:
      if not next_train_data_future.healty:
        raise Exception("some data gatherer worker has died")
      sleep(0.1)
    train_data = (next_hardware, next_circuit, next_train_data_future.data)

    return total_loss, total_cost_loss, total_vm_loss, vm_ratio, train_data
  

  def _validation(
    self,
    pheader: str,
    data_log: dict[str, list],
    save_path: str,
    train_cfg: TrainConfig,
    it: int,
    best_model: dict,
  ) -> float:
    print(f"{pheader} Running validation...", end='')
    circuits = train_cfg.validation_circuits
    # Optimize validation circuits and normalize costs
    all_res = self.optimize_mult(
      circuits=circuits,
      hardware=train_cfg.validation_hardware,
      n_workers=train_cfg.n_workers,
      devices=train_cfg.worker_devices,
    )
    val_cost = torch.tensor([res[1]/(c.n_gates_norm + 1) for res,c in zip(all_res, circuits)])
    vc_mean = val_cost.mean().item()
    data_log['val_cost'].append(vc_mean)
    print(f"{pheader}     vc={vc_mean:.4f}, ", end='')
    if best_model['val_cost'] is None:
      best_model = dict(val_cost=val_cost, vc_mean=vc_mean)
    else:
      p = ttest_ind(val_cost.numpy(), best_model['val_cost'].numpy(), equal_var=False)[1]
      if p < 0.2:
        if vc_mean < best_model['vc_mean']:
          print(f"better than prev {best_model['vc_mean']:.4f} with p={p:.3f}, ", end='')
          best_model = dict(val_cost=val_cost, vc_mean=vc_mean)
        else:
          print(f"worse than prev {best_model['vc_mean']:.4f} with p={p:.3f}, ", end='')
      else:
        print(f"not enough significance wrt prev={best_model['vc_mean']:.4f} p={p:.3f}, ", end='')
    # Save model checkpoint and validation data
    chkpt_name = f"checkpt_{it+1}_{int(vc_mean*1000)}.pt"
    print(f"saving as {chkpt_name}")
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, chkpt_name))
    with open(os.path.join(save_path, "train_data.json"), "w") as f:
      json.dump(data_log, f, indent=2)
    return best_model
