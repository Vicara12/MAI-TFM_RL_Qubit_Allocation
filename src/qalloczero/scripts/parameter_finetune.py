import torch
from utils.timer import Timer
from utils.allocutils import sol_cost
from sampler.randomcircuit import RandomCircuit
from qalloczero.models.predmodel import PredictionModel
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.ts_cpp import TSCppEngine




def linear_search():
  torch.manual_seed(42)
  ts_cfg_params = dict(
    # target_tree_size = [4, 8, 16, 32, 64 ,128, 256, 512, 1024, 2048, 4096, 8192],
    # target_tree_size = [1024, 2048, 4096, 8192],
    noise=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7],
    # dirichlet_alpha=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7],
    # discount_factor=[0, 0.05, 0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # action_sel_temp=[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7],
    # ucb_c1=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3, 3.5],
    # ucb_c2=[50, 500, 1_000, 5_000, 10_000, 15_000, 19_652, 20_000, 25_000, 30_000, 40_000],
    ucb_c1=[0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25],
    # ucb_c1=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
    # ucb_c2=[500, 1_000, 5_000, 10_000, 15_000, 19_652, 20_000, 25_000, 30_000, 40_000],
  )

  n_circuits = 16
  n_qubits = 16
  n_slices = 16
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  hardware = Hardware(
    core_capacities=core_caps,
    core_connectivity=torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  )
  # azero = AlphaZero(
  #   hardware,
  #   device='cpu',
  #   backend=AlphaZero.Backend.Cpp,
  # )
  azero = AlphaZero.load("trained/da_v4", device="cpu")
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  circuits = [sampler.sample() for i in range(n_circuits)]
  cfg_params = dict(
    target_tree_size=512,
    noise=0.25,
    dirichlet_alpha=0.25,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.15,
    ucb_c2=500,
  )
  # ignore_params = ['target_tree_size', 'action_sel_temp']
  ignore_params = ['action_sel_temp']
  results = dict()

  for (param, values) in ts_cfg_params.items():
    if param in ignore_params:
      print(f"[*] Ignoring {param}, as it is in ignore_params\n")
      continue
    print(f"[*] Testing for {param} with values cfg: {cfg_params}")
    results_param = {}
    for value in values:
      cfg_params[param] = value
      cfg = TSConfig(**cfg_params)
      with Timer.get('t'):
        exec_res = azero.optimize_mult(circuits, ts_cfg=cfg, hardware=hardware)
      get_mean = lambda idx: torch.mean(torch.tensor([float(res[idx]) for res in exec_res])).item()
      norm_cost = [res[1]/circ.n_gates_norm for (res,circ) in zip(exec_res, circuits)]
      avg_cost = sum(norm_cost)/len(norm_cost)
      avg_exp_nodes = get_mean(2)
      avg_expl_ratio = get_mean(3)
      results_param[value] = (avg_cost, avg_exp_nodes, avg_expl_ratio)
      print(f" - {value:.3f}: \tc={avg_cost:.2f} \ten={avg_exp_nodes:.2f} \ter={avg_expl_ratio:.2f} \tt={Timer.get('t').time:.2f}")
    best = min(results_param.items(), key=lambda x: x[1][0])
    print(f" + Using best: {best[0]}")
    results[param] = results_param
    cfg_params[param] = best[0]
    print()
  print(results)
    