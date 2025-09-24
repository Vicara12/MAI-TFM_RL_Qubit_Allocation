from typing import Tuple, Optional
from os import remove
from time import time
import torch
from dataclasses import dataclass
from qalloczero.alg.ts_cpp_engine.build.ts_cpp_engine import (
  TSEngine,
  TseOptConfig,
  TseTrainData
)



class TSCppEngine:
  def __init__(
    self,
    n_qubits: int,
    core_caps: torch.Tensor,
    core_cons: torch.Tensor
  ):
    self.cpp_engine = TSEngine(n_qubits, core_caps, core_cons)
  
  def load_model(self, name: str, model: torch.nn.Module):
    scripted_model = torch.jit.script(model)
    # Make model names unique to prevent clashes if run in parallel
    file_name = f"/tmp/model_{str(time()).replace('.','')}.pt"
    scripted_model.save(file_name)
    try:
      self.cpp_engine.load_model(name, file_name)
    finally:
      remove(file_name)
  
  def has_model(self, name: str) -> bool:
    return self.cpp_engine.has_model(name)
  
  def rm_model(self, name: str):
    return self.cpp_engine.rm_model(name)
  
  def optimize(
      self,
      slice_adjm: torch.Tensor,
      circuit_embs: torch.Tensor,
      alloc_steps: torch.Tensor,
      cfg: TseOptConfig,
      ret_train_data: bool
  ) -> Tuple[torch.Tensor, int, float, Optional[TseTrainData]]:
    return self.cpp_engine.optimize(
      slice_adjm,
      circuit_embs,
      alloc_steps,
      cfg,
      ret_train_data
    )