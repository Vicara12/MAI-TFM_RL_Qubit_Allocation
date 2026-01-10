from typing import Tuple, Optional
from os import remove
from time import time
import torch
from qalloczero.alg.ts import (TSEngine, TSConfig, TSTrainData)
from qalloczero.alg.ts_cpp_engine.build.ts_cpp_engine import (
  TSEngine as TSCppEngineInterface,
  TseOptConfig as TseCppOptConfig,
  TseTrainData as TseCppTrainData
)



class TSCppEngine:
  def __init__(
    self,
    device: str = "cpu",
  ):
    self.cpp_engine = TSCppEngineInterface(device)
  
  def load_model(self, model: torch.nn.Module):
    scripted_model = torch.jit.script(model)
    # Make model names unique to prevent clashes if run in parallel
    file_name = f"/tmp/model_{str(time()).replace('.','')}.pt"
    scripted_model.save(file_name)
    try:
      self.cpp_engine.load_model(file_name)
    finally:
      remove(file_name)
  
  def has_model(self) -> bool:
    return self.cpp_engine.has_model()
  
  def replace_model(self, model: torch.nn.Module):
    if self.has_model():
      self.cpp_engine.rm_model()
    self.load_model(model=model)
  
  def optimize(
    self,
    n_qubits: int,
    core_conns: torch.Tensor,
    core_caps: torch.Tensor,
    circuit_embs: torch.Tensor,
    next_inter: torch.Tensor,
    alloc_steps: torch.Tensor,
    cfg: TSConfig,
    ret_train_data: bool,
    verbose: bool,
  ) -> Tuple[torch.Tensor, int, float, Optional[TSTrainData]]:
    allocs, n_exp_nodes, expl_r, tdata = self.cpp_engine.optimize(
      n_qubits,
      core_conns,
      core_caps,
      circuit_embs,
      next_inter,
      alloc_steps,
      TSCppEngine._convert_cfg(cfg),
      ret_train_data,
      verbose,
    )
    return allocs, n_exp_nodes, expl_r, TSCppEngine._convert_train_data(tdata)
  
  @staticmethod
  def _convert_cfg(cfg: TSConfig) -> TseCppOptConfig:
    new_cfg = TseCppOptConfig()
    new_cfg.target_tree_size = cfg.target_tree_size
    new_cfg.noise = cfg.noise
    new_cfg.dirichlet_alpha = cfg.dirichlet_alpha
    new_cfg.discount_factor = cfg.discount_factor
    new_cfg.action_sel_temp = cfg.action_sel_temp
    new_cfg.ucb_c1 = cfg.ucb_c1
    new_cfg.ucb_c2 = cfg.ucb_c2
    return new_cfg
  
  @staticmethod
  def _convert_train_data(
    tdata: Optional[TseCppTrainData]
  ) -> Optional[TSTrainData]:
    if tdata is None:
      return None
    return TSTrainData(
      qubits = tdata.qubits,
      prev_allocs = tdata.prev_allocs,
      curr_allocs = tdata.curr_allocs,
      core_caps = tdata.core_caps,
      slice_idx = tdata.slice_idx,
      logits = tdata.logits,
      value = tdata.value,
    )