import torch
from utils.timer import Timer
from utils.allocutils import sol_cost
from sampler.randomcircuit import RandomCircuit
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.predmodel import PredictionModel
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.ts_cpp import TSCppEngine



def finetune():
  torch.manual_seed(42)
  n_qubits = 64
  n_slices = 32
  core_caps = torch.tensor([8]*8, dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_connectivity = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  pred_model = PredictionModel(
    n_qubits=n_qubits,
    n_cores=core_connectivity.shape[0],
    core_connectivity=core_connectivity,
    number_emb_size=4,
    n_heads=4,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  circuit = sampler.sample()
  print(f"{circuit.slice_gates = }")
  encoder = CircuitEncoder(n_qubits=n_qubits, n_heads=4, n_layers=4)
  encoder.eval()
  embs = encoder(circuit.adj_matrices.unsqueeze(0)).squeeze(0)
  cpp_engine = TSCppEngine(
    n_qubits,
    n_cores,
    device="cuda",
  )
  cpp_engine.load_model(pred_model)
  times = []
  costs = []
  timer = Timer.get('t')
  for tts in [8, 16, 32, 64, 128, 256, 512, 1024]:
    torch.manual_seed(42)
    print(f"Optimizing with tts {tts}")
    with timer:
      res_py = cpp_engine.optimize(
        slice_adjm=circuit.adj_matrices,
        circuit_embs=embs,
        alloc_steps=circuit.alloc_steps,
        cfg=TSConfig(target_tree_size=tts),
        ret_train_data=False,
      )
    costs.append(sol_cost(res_py[0], core_connectivity))
    times.append(timer.time)
    timer.reset()
  print(f"{times = }")
  print(f"{costs = }")
  return times, costs


def grid_search():
  all_times = []
  all_costs = []
  try:
    for i in range(100):
      t, c = finetune()
      all_times.append(t)
      all_costs.append(c)
  except KeyboardInterrupt:
    pass
  print(f"{all_times = }")
  print(f"{all_costs = }")