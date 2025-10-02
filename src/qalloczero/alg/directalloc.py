import os
import json
import torch
import warnings
from typing import Self, Tuple, Optional
from dataclasses import dataclass
from utils.customtypes import Circuit, Hardware
from utils.allocutils import sol_cost
from qalloczero.alg.ts import ModelConfigs
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.predmodel import PredictionModel



@dataclass
class DAConfig:
  noise: float = 0.0
  mask_invalid: bool = True
  greedy: bool = True


class DirectAllocator:

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
    if cfg.mask_invalid:
      pol[~valid_cores] = 0
    # Add exploration noise to the priors
    if cfg.noise != 0:
      noise = torch.abs(torch.randn(valid_cores.shape))
      pol[valid_cores] = (1 - cfg.noise)*pol[valid_cores] + cfg.noise*noise
    sum_pol = sum(pol)
    if sum_pol < 1e-5:
      pol = torch.zeros_like(pol)
      n_valid_cores = sum(valid_cores)
      assert n_valid_cores > 0, "No valid allocation possible"
      pol[valid_cores] = 1/n_valid_cores
    else:
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
    ret_train_data: bool
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
      aux_data_shape = [alloc_steps.shape[0], self.hardware.n_cores]
      all_actions = torch.empty([alloc_steps.shape[0]], dtype=torch.int, device=self.device)
      all_probs = torch.empty(aux_data_shape, device=self.device)
      all_valid_cores = torch.empty_like(aux_data_shape, dtype=torch.bool, device=self.device)
    prev_slice = -1
    for step, (slice_idx, qubit0, qubit1, _) in enumerate(alloc_steps):
      if prev_slice != slice_idx:
        prev_core_allocs = core_allocs
        core_allocs = self.hardware.n_cores * torch.ones_like(core_allocs)
        core_caps = core_caps_orig.clone()
      pol, _ = self.pred_model(
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
        all_actions[step] = action
        all_probs[step] = pol
        all_valid_cores[step] = valid_cores
      core_caps[action] = max(0, core_caps[action] - n_qubits)
      prev_slice = slice_idx
    if ret_train_data:
      return all_actions, all_probs, all_valid_cores
  

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
    )
    cost = sol_cost(allocations=allocations, core_con=self.hardware.core_connectivity)
    return allocations, cost
    