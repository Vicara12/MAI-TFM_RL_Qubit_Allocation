from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch



@dataclass
class TSConfig:
    target_tree_size: int = 1024
    noise: float = 0.25
    dirichlet_alpha: float = 0.3
    discount_factor: float = 1.0
    action_sel_temp: float = 0.0
    ucb_c1: float = 1.25
    ucb_c2: float = 19652


@dataclass
class TSTrainData:
    qubits: torch.Tensor
    prev_allocs: torch.Tensor
    curr_allocs: torch.Tensor
    core_caps: torch.Tensor
    slice_idx: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor


@dataclass
class ModelConfigs:
    layers: List[int] = field(default_factory=lambda: [16, 32, 64])


class TSEngine(ABC):
    ''' Abstract class for performing tree search with DL heuristics.
    '''

    @abstractmethod
    def load_model(self, model: torch.nn.Module):
        raise NotImplementedError(
            f"load_model not implemented for class {self.__class__.__name__}")
    
    @abstractmethod
    def has_model(self) -> bool:
        raise NotImplementedError(
            f"has_model not implemented for class {self.__class__.__name__}")
    
    @abstractmethod
    def replace_model(self, model: torch.nn.Module):
        raise NotImplementedError(
            f"replace_model not implemented for class {self.__class__.__name__}")
        
    @abstractmethod
    def optimize(
      self,
      n_qubits: int,
      core_conns: torch.Tensor,
      core_caps: torch.Tensor,
      circuit_embs: torch.Tensor,
      alloc_steps: torch.Tensor,
      cfg: TSConfig,
      ret_train_data: bool,
      verbose: bool = False
  ) -> Tuple[torch.Tensor, int, float, Optional[TSTrainData]]:
        raise NotImplementedError(
            f"optimize not implemented for class {self.__class__.__name__}")