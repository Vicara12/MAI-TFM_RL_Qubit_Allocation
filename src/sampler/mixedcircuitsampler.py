import random
from utils.customtypes import Circuit
from sampler.circuitsampler import CircuitSampler


class MixedCircuitSampler(CircuitSampler):
  '''
  Samples from a list of circuit samplers with certain probability.
  '''

  def __init__(self, num_lq: int, samplers: list[tuple[float, CircuitSampler]]):
    self.num_lq_ = num_lq
    self.probs = [p for (p, _) in samplers]
    pmass = sum(self.probs)
    if pmass != 1:
      raise ValueError(f"Probs of all samplers must add up to 1, got {pmass}: {self.probs}")
    self.samplers = [s for (_, s) in samplers]
  
  def sample(self) -> Circuit:
    sampler = random.choices(self.samplers, weights=self.probs, k=1)[0]
    sampler.num_lq = self.num_lq
    return sampler.sample()

  def __str__(self) -> str:
    raise f"MixedCircuitSampler(num_lq={self.num_lq}, samplers={list(zip(self.probs, self.samplers))})"