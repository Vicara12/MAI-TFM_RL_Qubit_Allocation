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



def linear_search():
  torch.manual_seed(42)
  ts_cfg_params = dict(
    target_tree_size = [4, 8, 16, 32, 64 ,128, 256],
    noise=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7],
    dirichlet_alpha=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7],
    discount_factor=[0, 0.05, 0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    action_sel_temp=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7],
    # ucb_c1=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3, 3.5],
    # ucb_c2=[50, 500, 1_000, 5_000, 10_000, 15_000, 19_652, 20_000, 25_000, 30_000, 40_000],
    ucb_c1=[0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25],
    ucb_c2=[500, 1_000, 5_000, 10_000, 15_000, 19_652, 20_000, 25_000, 30_000, 40_000],
  )

  n_circuits = 8
  n_qubits = 16
  n_slices = 32
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  hardware = Hardware(
    core_capacities=core_caps,
    core_connectivity=torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  )
  azero = AlphaZero(
    hardware,
    device='cpu',
    backend=AlphaZero.Backend.Cpp,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  circuits = [sampler.sample() for i in range(n_circuits)]
  best_params = dict(
    target_tree_size=64,
    noise=0.00,
    dirichlet_alpha=0.0,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.025,
    # ucb_c2=500,
  )
  results = dict()

  for (param, values) in ts_cfg_params.items():
    if param in best_params.keys():
      print(f"[*] Ignoring {param}, as it is in best_params\n")
      continue
    print(f"[*] Testing for {param} with values cfg: {best_params}")
    results_param = {}
    for value in values:
      cfg = TSConfig(**(best_params | {param:value}))
      with Timer.get('t'):
        exec_res = azero.optimize_mult(circuits, cfg)
      get_mean = lambda idx: torch.mean(torch.tensor([float(res[idx]) for res in exec_res])).item()
      avg_cost = get_mean(1)
      avg_exp_nodes = get_mean(2)
      avg_expl_ratio = get_mean(3)
      results_param[value] = (avg_cost, avg_exp_nodes, avg_expl_ratio)
      print(f" - {value:.3f}: \tc={avg_cost:.2f} \ten={avg_exp_nodes:.2f} \ter={avg_expl_ratio:.2f} \tt={Timer.get('t').time:.2f}")
    best = min(results_param.items(), key=lambda x: x[1][0])
    print(f" + Using best: {best[0]}")
    results[param] = results_param
    if param not in best_params.keys():
      best_params[param] = best[0]
    else:
      print(f" + Ignoring best in future search")
    print()
  print(results)
    