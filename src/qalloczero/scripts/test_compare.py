import torch
import json
import pandas as pd
from utils.timer import Timer
from utils.customtypes import Hardware, Circuit
from sampler.randomcircuit import RandomCircuit
from utils.plotter import drawCircuit
from utils.allocutils import check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap, get_all_checkpoints
from qalloczero.alg.ts import TSConfig
# from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.directalloc import DirectAllocator
from russo.tests.hungarian import HQA


def validate():
  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 32
  n_circuits = 16
  core_caps = torch.tensor([4]*4, dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  algos = dict(
    hqa = HQA(lookahead=True, verbose=True),
    da_seq  = DirectAllocator.load("trained/da_v13_ft",    device="cuda", checkpoint=9475).set_mode(DirectAllocator.Mode.Sequential),
    da_par  = DirectAllocator.load("trained/da_v13_ft",    device="cuda", checkpoint=9475).set_mode(DirectAllocator.Mode.Parallel),
    
    # da_seq_v2 = DirectAllocator.load("trained/da_v2", device="cpu").set_mode(DirectAllocator.Mode.Sequential),
    # da_seq_v4 = DirectAllocator.load("trained/da_v4", device="cpu").set_mode(DirectAllocator.Mode.Sequential),
    # da_seq_v5 = DirectAllocator.load("trained/da_v6", device="cpu").set_mode(DirectAllocator.Mode.Sequential),

    # da_par_v  = DirectAllocator.load("trained/da_v7"   , device="cpu", checkpoint=175).set_mode(DirectAllocator.Mode.Parallel),
    # da_par_v2 = DirectAllocator.load("trained/da_v2", device="cpu").set_mode(DirectAllocator.Mode.Parallel),
    # da_par_v4 = DirectAllocator.load("trained/da_v4", device="cpu").set_mode(DirectAllocator.Mode.Parallel),
    # da_par_v5 = DirectAllocator.load("trained/da_v5", device="cpu").set_mode(DirectAllocator.Mode.Parallel),

    # az_v  = AlphaZero.load("trained/da",    device="cpu"),
    # az_v2 = AlphaZero.load("trained/da_v2", device="cpu"),
    # az_v4 = AlphaZero.load("trained/da_v4", device="cpu"),

    # da_sequential = DirectAllocator.load("trained/da_v6", device="cuda", checkpoint=1000).set_mode(DirectAllocator.Mode.Sequential),
    # da_parallel   = DirectAllocator.load("trained/da_v6", device="cuda", checkpoint=1000).set_mode(DirectAllocator.Mode.Parallel),

  )
  cfg = TSConfig(
    target_tree_size=512,
    noise=0.2,
    dirichlet_alpha=1.0,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.125,
    ucb_c2=500,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices, reflow=False)
  circuits = [sampler.sample() for _ in range(n_circuits)]
  for (name, algo) in algos.items():
    print(f"[*] Optimizing {name}")
    with Timer.get('t'):
      if isinstance(algo, DirectAllocator) or isinstance(algo, HQA):
        results = []
        for circ in circuits:
          results.append(algo.optimize(circ, hardware=hardware, verbose=True))
      elif isinstance(algo, AlphaZero):
        results = algo.optimize_mult(circuits, cfg, hardware=hardware, verbose=True)
      else:
        raise Exception("Unrecognized algorithm type")
    norm_res = torch.tensor([res[1]/circ.n_gates_norm for (res, circ) in zip(results,circuits)])
    for (res, circuit) in zip(results, circuits):
      check_sanity(allocs=res[0], circuit=circuit, hardware=hardware)
    norm_swaps = [
      count_swaps(swaps_from_alloc(res[0], n_cores))/circ.n_gates_norm for (res, circ) in zip(results,circuits)
    ]
    print(f" + t={Timer.get('t').time:.2f}s avg_cost={norm_res.mean().item():.4f} ({norm_res.std().item():.2f}) avg_swaps={sum(norm_swaps)/len(norm_swaps):.4f}")


def benchmark():

  # [*] Optimizing with hqa
  #  + qft: t=873.88s cost=1149.0 (0.23)
  #  + graph_state: t=1278.25s cost=1485.0 (0.61)
  #  + deutsch_jozsa: t=40.84s cost=36.0 (0.36)

  circuit_names = [
    "qft", # Exact
    # "quantum_volume",
    "graph_state", # Exact
    # "drapper_adder",
    # "cuccaro_adder", # A bit over
    # "qnn",
    "deutsch_jozsa", # Exact
  ]
  circuits = {name: Circuit.from_qasm(f'circuits/{name}_100.qasm', 100) for name in circuit_names}
  # A2A configuration
  hardware = Hardware(
    core_capacities=torch.tensor([10]*10),
    core_connectivity=(torch.ones(size=(10,10)) - torch.eye(10)),
  )


  algos = dict(
    # hqa = HQA(lookahead=True, verbose=True),
    da_sequential = DirectAllocator.load("trained/da_v13_ft", device="cuda", checkpoint=9475).set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load("trained/da_v13_ft", device="cuda", checkpoint=9475).set_mode(DirectAllocator.Mode.Parallel),
    # azero =               AlphaZero.load("trained/az", device="cpu"),
    # da_azero = DirectAllocator.load("trained/azero", device="cuda"),
    # azero_azero =    AlphaZero.load("trained/azero", device="cpu"),
  )
  cfg = TSConfig(
    target_tree_size=512,
    noise=0.2,
    dirichlet_alpha=1.0,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.125,
    ucb_c2=500,
  )

  for (name, algo) in algos.items():
    print(f"[*] Optimizing with {name}")
    for cname, circ in circuits.items():
      with Timer.get('t'):
        if isinstance(algo, HQA):
          cs, cost = algo.optimize(circ, hardware=hardware)
        elif isinstance(algo, DirectAllocator):
          allocs, cost = algo.optimize(circ, hardware=hardware, verbose=True)
        elif isinstance(algo, AlphaZero):
          allocs, cost, _, er = algo.optimize(circ, cfg, hardware=hardware, verbose=True)
        else:
          raise Exception("Unrecognized algorithm type")
      print(f" + {cname}: t={Timer.get('t').time:.2f}s cost={cost} ({cost/(circ.n_gates_norm+1):.2f})")


def compare_w_sota():
  n_qubits=50
  base_res = pd.read_csv(f'data/sota_cost_{n_qubits}.csv', index_col="circuit")
  base_times = pd.read_csv(f'data/sota_time_{n_qubits}.csv', index_col="circuit")

  with open(f'data/all_{n_qubits}.json', 'r') as f:
    data = json.load(f)

  n_qubits = data['n_qubits']
  circuit_slices = data['circuits']
  circuits = {}
  for (name, slices) in circuit_slices.items():
    circuits[name] = Circuit(slice_gates=slices, n_qubits=n_qubits)
  
  algos = dict(
    da_sequential = DirectAllocator.load("trained/da_v13_ft", device="cuda", checkpoint=9475).set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load("trained/da_v13_ft", device="cuda", checkpoint=9475).set_mode(DirectAllocator.Mode.Parallel),
  )

  n_cores = n_qubits//10
  hardware = Hardware(
    core_capacities=torch.tensor([10]*n_cores),
    core_connectivity=(torch.ones(size=(n_cores,n_cores)) - torch.eye(n_cores)),
  )

  my_results = {}
  my_times = {}

  for (name, algo) in algos.items():
    my_results[name] = {}
    my_times[name] = {}
    print(f"[*] Optimizing with {name}")
    for cname, circ in circuits.items():
      with Timer.get('t'):
        if isinstance(algo, HQA):
          cs, cost = algo.optimize(circ, hardware=hardware)
        elif isinstance(algo, DirectAllocator):
          allocs, cost = algo.optimize(circ, hardware=hardware, verbose=True)
        elif isinstance(algo, AlphaZero):
          pass
          # allocs, cost, _, er = algo.optimize(circ, cfg, hardware=hardware, verbose=True)
        else:
          raise Exception("Unrecognized algorithm type")
      print(f" + {cname}: t={Timer.get('t').time:.2f}s cost={cost} ({cost/(circ.n_gates_norm+1):.2f})")
      my_results[name][cname] = cost
      my_times[name][cname] = Timer.get('t').time
  
  print(my_results)
  print(my_times)

  my_results = pd.DataFrame.from_dict(my_results, orient="index").T
  all_res = pd.concat([base_res, my_results], axis=1)
  all_res.to_csv(f'data/my_cost_{n_qubits}_.csv', index=True)
  my_times = pd.DataFrame.from_dict(my_times, orient="index").T
  all_times = pd.concat([base_times, my_times], axis=1)
  all_times.to_csv(f'data/my_time_{n_qubits}_.csv', index=True)
