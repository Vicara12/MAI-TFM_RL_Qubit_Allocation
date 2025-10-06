import os
import json
from scipy import stats
import torch
from copy import deepcopy
import warnings
from itertools import chain
from typing import Self, Tuple, List, Optional
from dataclasses import dataclass
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost
from utils.timer import Timer
from utils.gradient_tools import print_grad
from sampler.circuitsampler import CircuitSampler
from qalloczero.alg.ts import ModelConfigs
from qalloczero.models.enccircuit import CircuitEncoder
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
    validation_size: int
    initial_noise: float
    noise_decrease_factor: int
    sampler: CircuitSampler
    lr: float
    invalid_move_penalty: float
    repl_significance: float
    print_grad_each: Optional[int] = None


  def __init__(
    self,
    hardware: Hardware,
    device: str = "cpu",
    model_cfg: ModelConfigs = ModelConfigs()
  ):
    self.hardware = hardware
    self.core_conn = hardware.core_connectivity.to(device)
    self.device = device
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
    self.pred_model.to(device)
  

  def save(self, path: str, overwrite: bool = False):
    params = dict(
      core_caps=self.hardware.core_capacities.tolist(),
      core_conns=self.hardware.core_connectivity.tolist(),
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
  def load(path: str, device: str = "cuda") -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "optimizer_conf.json"), "r") as f:
      params = json.load(f)
    hardware = Hardware(torch.tensor(params["core_caps"]), torch.tensor(params["core_conns"]))
    loaded = DirectAllocator(hardware=hardware, device=device)
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
    return loaded
  

  def _sample_action(
    self,
    pol: torch.Tensor,
    core_caps: torch.Tensor,
    n_qubits: int,
    cfg: DAConfig,
  ) -> Tuple[int, torch.Tensor, torch.Tensor]:
    pol = pol.squeeze(0)
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
    pol /= sum_pol
    core = pol.argmax().item() if cfg.greedy else torch.distributions.Categorical(pol).sample()
    return core, pol, valid_cores
  

  def _allocate(
    self,
    allocations: torch.Tensor,
    adj_mats: torch.Tensor,
    circ_embs: torch.Tensor,
    alloc_steps: torch.Tensor,
    cfg: DAConfig,
    ret_train_data: bool,
    pred_mod: torch.nn.Module,
  ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    core_caps_orig = self.hardware.core_capacities.to(self.device)
    core_allocs = self.hardware.n_cores * torch.ones(
      [self.hardware.n_qubits],
      dtype=torch.long,
      device=self.device,
    )
    prev_core_allocs = None
    core_caps = None
    if ret_train_data:
      all_actions = []
      all_probs = []
      all_valid_cores = []
    prev_slice = -1
    for (slice_idx, qubit0, qubit1, _) in alloc_steps:
      if prev_slice != slice_idx:
        prev_core_allocs = core_allocs
        core_allocs = self.hardware.n_cores * torch.ones_like(core_allocs)
        core_caps = core_caps_orig.clone()
      pol, _ = pred_mod(
        qubits=torch.tensor([qubit0, qubit1], dtype=torch.int, device=self.device).unsqueeze(0),
        prev_core_allocs=prev_core_allocs.unsqueeze(0),
        current_core_allocs=core_allocs.unsqueeze(0),
        core_capacities=core_caps.unsqueeze(0),
        core_connectivity=self.core_conn,
        circuit_emb=circ_embs[:,slice_idx],
        slice_adj_mat=adj_mats[:,slice_idx],
      )
      n_qubits = (1 if qubit1 == -1 else 2)
      action, pol, valid_cores = self._sample_action(
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

        all_actions.append(action)
        all_probs.append(pol)
        all_valid_cores.append(valid_cores)
      core_caps[action] = max(0, core_caps[action] - n_qubits)
      prev_slice = slice_idx
    if ret_train_data:
      return torch.tensor(all_actions), torch.stack(all_probs), torch.stack(all_valid_cores)
  

  def optimize(
    self,
    circuit: Circuit,
    cfg: DAConfig = DAConfig(),
  ) -> Tuple[torch.Tensor, float]:
    if circuit.n_qubits != self.hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {self.hardware.n_qubits} != {circuit.n_qubits}"
      ))
    adj_mats = circuit.adj_matrices.unsqueeze(0).to(self.device)
    circ_embs = self.circ_enc(adj_mats)
    allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
    self._allocate(
      allocations=allocations,
      adj_mats=adj_mats,
      circ_embs=circ_embs,
      alloc_steps=circuit.alloc_steps,
      cfg=cfg,
      ret_train_data=False,
      pred_mod=self.pred_model,
    )
    cost = sol_cost(allocations=allocations, core_con=self.hardware.core_connectivity)
    return allocations, cost
  

  def train(
    self,
    train_cfg: TrainConfig,
  ):
    self.iter_timer = Timer.get("_train_iter_timer")
    self.iter_timer.reset()
    optimizer = torch.optim.Adam(
      chain(self.circ_enc.parameters(), self.pred_model.parameters()), lr=train_cfg.lr
    )
    circ_enc_bl, pred_model_bl = self._clone_models()
    opt_cfg = DAConfig(
      noise=train_cfg.initial_noise,
      mask_invalid=False,
      greedy=False,
    )
    self.pgrad_counter = 1

    try:
      for iter in range(train_cfg.train_iters):
        self.iter_timer.start()
        frac_valid_moves, cost_loss, vm_loss = self._train_batch(
          iter=iter,
          circ_enc_bl=circ_enc_bl,
          pred_model_bl=pred_model_bl,
          optimizer=optimizer,
          opt_cfg=opt_cfg,
          train_cfg=train_cfg,
        )
        opt_cfg.noise *= train_cfg.noise_decrease_factor

        print(f"\033[2K\r[{iter + 1}/{train_cfg.train_iters}] Running validation...", end='')
        with torch.no_grad():
          cost, cost_bl = self._validation(
            circ_enc_bl=circ_enc_bl,
            pred_model_bl=pred_model_bl,
            train_cfg=train_cfg,
          )

        if cost == cost_bl:
          p_value = 0.5
        else:
          _, p_value = stats.ttest_rel(cost, cost_bl, alternative='less')
        replace = p_value < train_cfg.repl_significance
        if replace:
          circ_enc_bl, pred_model_bl = self._clone_models()

        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - iter - 1)
        avg = lambda l: sum(l)/len(l)
        print((
          f"\033[2K\r[{iter + 1}/{train_cfg.train_iters}] "
          f"loss={cost_loss + vm_loss:.3f} (c={cost_loss:.3f},vm={vm_loss:.3f}) \t"
          f"vm={frac_valid_moves:.3f} "
          f"cost={avg(cost):.3f} cost_bl={avg(cost_bl):.3f} pv={p_value:.2f} "
          f"{'REPLACED' if replace else '        '} "
          f"t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left)"
        ))
        if train_cfg.print_grad_each is not None and self.pgrad_counter == train_cfg.print_grad_each:
          print(f"\n[+] Gradient information for prediction model:")
          print_grad(self.pred_model)
          print(f"\n[+] Gradient information for circuit encoder:")
          print_grad(self.circ_enc)
          self.pgrad_counter = 1
        else:
          self.pgrad_counter += 1


    except KeyboardInterrupt as e:
      if 'y' not in input('\nGraceful shutdown? [y/n]: ').lower():
        raise e
  

  def _train_batch(
    self,
    iter: int,
    circ_enc_bl: torch.nn.Module,
    pred_model_bl: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    opt_cfg: DAConfig,
    train_cfg: TrainConfig,
  ) -> float:
    self.circ_enc.train()
    self.pred_model.train()
    circuits = [train_cfg.sampler.sample() for _ in range(train_cfg.batch_size)]
    print(f"\033[2K\r[{iter + 1}/{train_cfg.train_iters}] Obtaining baselines...", end='')
    with torch.no_grad():
      costs_bl = self._get_baseline(circuits, opt_cfg, circ_enc_bl, pred_model_bl)
    adj_mats = torch.stack([c.adj_matrices.to(self.device) for c in circuits], dim=0)
    circ_embs = self.circ_enc(adj_mats)
    cost_loss = 0
    valid_move_loss = 0
    n_valid_moves = 0
    total_moves = sum([c.n_steps for c in circuits])
    for batch_i, circuit in enumerate(circuits):
      print(
        f"\033[2K\r[{iter + 1}/{train_cfg.train_iters}] Optimizing {batch_i + 1}/{train_cfg.batch_size}",
        end=''
      )
      allocations = torch.empty([circuits[0].n_slices, circuits[0].n_qubits], dtype=torch.int)
      actions, probs, valid_cores = self._allocate(
        allocations=allocations,
        adj_mats=adj_mats[batch_i].unsqueeze(0),
        circ_embs=circ_embs[batch_i].unsqueeze(0),
        alloc_steps=circuit.alloc_steps,
        cfg=opt_cfg,
        ret_train_data=True,
        pred_mod=self.pred_model,
      )
      cost = sol_cost(allocations=allocations, core_con=self.hardware.core_connectivity)
      over_cost = (cost - costs_bl[batch_i]) / (circuit.n_gates_norm + 1)
      valid_moves = valid_cores[torch.arange(valid_cores.shape[0]), actions]
      n_valid_moves += torch.sum(valid_moves).item()
      action_probs = probs[torch.arange(probs.shape[0]), actions]
      # This loss tries to maximize the probability of actions that resulted in reduced cost wrt the
      # baseline and minimize those that resulted in worse cost wrt to the baseline
      cost_loss += over_cost * torch.sum(action_probs[valid_moves])
      # This loss tries to maximize the number of valid moves (vm) the network does
      valid_move_loss += torch.sum(action_probs[~valid_moves])
    valid_move_loss *= train_cfg.invalid_move_penalty
    # loss = cost_loss + valid_move_loss
    loss = cost_loss # TODO remove and set loss as sum
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return n_valid_moves/total_moves, cost_loss.item(), valid_move_loss.item()
  

  def _validation(
    self,
    circ_enc_bl: torch.nn.Module,
    pred_model_bl: torch.nn.Module,
    train_cfg: TrainConfig,
  ) -> Tuple[List[float], List[float]]:
    self.circ_enc.eval()
    self.pred_model.eval()
    da_cfg = DAConfig()
    circuits = [train_cfg.sampler.sample() for _ in range(train_cfg.validation_size)]
    new_seed = torch.randint(0, 1_000_000, []).item()
    torch.manual_seed(new_seed)
    cost = [self.optimize(circ, da_cfg)[1] for circ in circuits]
    torch.manual_seed(new_seed)
    costs_bl = self._get_baseline(circuits, da_cfg, circ_enc_bl, pred_model_bl)
    norm_item = lambda x: x[1]/circuits[x[0]].n_gates_norm
    return list(map(norm_item, enumerate(cost))), list(map(norm_item, enumerate(costs_bl)))


  def _get_baseline(
    self,
    circuits: List[Circuit],
    cfg: DAConfig,
    circ_enc_bl: torch.nn.Module,
    pred_model_bl: torch.nn.Module,
  ) -> List[float]:
    # Get baseline cost by swapping baseline and train models
    self.circ_enc, circ_enc_bl = circ_enc_bl, self.circ_enc
    self.pred_model, pred_model_bl = pred_model_bl, self.pred_model
    cost_bl = [self.optimize(circ, cfg)[1] for circ in circuits]
    self.circ_enc, circ_enc_bl = circ_enc_bl, self.circ_enc
    self.pred_model, pred_model_bl = pred_model_bl, self.pred_model
    return cost_bl
  

  def _clone_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
    circ_enc = deepcopy(self.circ_enc)
    pred_model = deepcopy(self.pred_model)
    circ_enc.eval()
    pred_model.eval()
    return circ_enc, pred_model