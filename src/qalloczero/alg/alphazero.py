import torch
from enum import Enum
from typing import Tuple
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
    sampler: CircuitSampler
    lr: float
    v_weight: float
    logit_weight: float


  def __init__(
      self,
      hardware: Hardware,
      verbose: bool = False,
      device: str = "cpu",
      backend: Backend = Backend.Cpp,
      model_cfg: ModelConfigs = ModelConfigs()
    ):
    self.n_qubits = hardware.n_qubits
    self.core_conns = hardware.core_connectivity
    self.backend = backend.value(
      n_qubits=hardware.n_qubits,
      core_caps=hardware.core_capacities,
      core_conns=hardware.core_connectivity,
      verbose=verbose,
      device=device,
    )
    self.circ_enc = CircuitEncoder(
      n_qubits=hardware.n_qubits,
      n_heads=model_cfg.ce_nheads,
      n_layers=model_cfg.ce_nlayers,
    )
    self.pred_model = PredictionModel(
      n_qubits=self.n_qubits,
      n_cores=hardware.n_cores,
      core_connectivity=hardware.core_connectivity,
      number_emb_size=model_cfg.pm_nemb_sz,
      n_heads=model_cfg.ce_nheads,
    )
    self.backend.load_model('pred_model', self.pred_model)

  def optimize(
      self,
      circuit: Circuit,
      ts_cfg: TSConfig
    ) -> Tuple[torch.Tensor, float, int, float]:
    if circuit.n_qubits != self.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {self.n_qubits} != {circuit.n_qubits}"
      ))
    circ_embs = self.circ_enc(circuit.adj_matrices.unsqueeze(0)).squeeze(0)
    allocations, exp_nodes, expl_ratio, _ = self.backend.optimize(
      slice_adjm=circuit.adj_matrices,
      circuit_embs=circ_embs,
      alloc_steps=circuit.alloc_steps,
      cfg=ts_cfg,
      ret_train_data=False
    )
    cost = solutionCost(allocations, self.core_conns)
    return allocations, cost, exp_nodes, expl_ratio