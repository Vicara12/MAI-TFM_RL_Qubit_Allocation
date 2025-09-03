from os import remove
from time import time
import torch
import qalloczero.alg.ts_cpp_engine.build.ts_cpp_engine as ts


class TSCppEngine:
  def __init__(self):
    self.cpp_engine = ts.TSEngine()
  
  def load_model(self, model: torch.nn.Module):
    scripted_model = torch.jit.script(model)
    # Make model names unique to prevent clashes if run in parallel
    file_name = f"/tmp/model_{str(time()).replace('.','')}.pt"
    scripted_model.save(file_name)
    self.cpp_engine.load_model(file_name)
    remove(file_name)
  
  def test_forward(self, *kwargs):
    return self.cpp_engine.test_fwd(*kwargs)