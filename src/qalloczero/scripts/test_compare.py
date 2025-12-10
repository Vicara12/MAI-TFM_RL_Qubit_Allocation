import torch
import json
import pandas as pd
from utils.timer import Timer
from utils.customtypes import Hardware, Circuit
from sampler.randomcircuit import RandomCircuit
from utils.plotter import drawCircuit
from utils.allocutils import check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap, get_all_checkpoints
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.alphazero import AlphaZero
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
    da_seq  = DirectAllocator.load("trained/da_v7",    device="cuda", checkpoint=-1).set_mode(DirectAllocator.Mode.Sequential),
    da_par  = DirectAllocator.load("trained/da_v7",    device="cuda", checkpoint=-1).set_mode(DirectAllocator.Mode.Parallel),
    
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

  # for (name, circ) in circuits.items():
  #   drawCircuit(circ.slice_gates[:60], circ.n_qubits, name, save_name=f'{name}.svg')
  # return

  # path = "trained/da_v6"
  # checks = sorted(list(get_all_checkpoints(path).keys()))[::4]

  # alogs_seq = {f"da_seq_{i}": DirectAllocator.load("trained/da_v6", device="cuda", checkpoint=i).set_mode(DirectAllocator.Mode.Sequential) for i in checks}
  # alogs_par = {f"da_seq_{i}": DirectAllocator.load("trained/da_v6", device="cuda", checkpoint=i).set_mode(DirectAllocator.Mode.Parallel) for i in checks}

  # algos = alogs_par

  algos = dict(
    # hqa = HQA(lookahead=True, verbose=True),
    da_sequential = DirectAllocator.load("trained/da_v7", device="cuda", checkpoint=-1).set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load("trained/da_v7", device="cuda", checkpoint=-1).set_mode(DirectAllocator.Mode.Parallel),
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
  base_res = pd.read_csv('data/sota_results_50.csv', index_col="circuit")

  with open('data/all_50.json', 'r') as f:
    data = json.load(f)

  n_qubits = data['n_qubits']
  circuit_slices = data['circuits']
  circuits = {}
  for (name, slices) in circuit_slices.items():
    circuits[name] = Circuit(slice_gates=slices, n_qubits=n_qubits)
  
  algos = dict(
    da_sequential = DirectAllocator.load("trained/da_v7", device="cuda", checkpoint=-1).set_mode(DirectAllocator.Mode.Sequential),
    # da_parallel   = DirectAllocator.load("trained/da_v10", device="cuda", checkpoint=-1).set_mode(DirectAllocator.Mode.Parallel),
  )

  n_cores = 5
  hardware = Hardware(
    core_capacities=torch.tensor([10]*n_cores),
    core_connectivity=(torch.ones(size=(n_cores,n_cores)) - torch.eye(n_cores)),
  )

  my_results = {}

  # my_results = {'da_sequential': {'random0': 249.0, 'random1': 239.0, 'random2': 313.0, 'random3': 287.0, 'random4': 288.0, 'random5': 258.0, 'random6': 275.0, 'random7': 240.0, 'random8': 288.0, 'random9': 287.0, 'random10': 260.0, 'random11': 284.0, 'random12': 265.0, 'random13': 260.0, 'random14': 306.0, 'random15': 260.0, 'random16': 274.0, 'random17': 292.0, 'random18': 316.0, 'random19': 279.0, 'random20': 250.0, 'random21': 274.0, 'random22': 318.0, 'random23': 261.0, 'random24': 290.0, 'random25': 261.0, 'random26': 262.0, 'random27': 269.0, 'random28': 295.0, 'random29': 279.0, 'random30': 256.0, 'random31': 283.0, 'random32': 278.0, 'random33': 257.0, 'random34': 283.0, 'random35': 271.0, 'random36': 267.0, 'random37': 267.0, 'random38': 277.0, 'random39': 265.0, 'random40': 237.0, 'random41': 298.0, 'random42': 328.0, 'random43': 267.0, 'random44': 289.0, 'random45': 262.0, 'random46': 272.0, 'random47': 247.0, 'random48': 269.0, 'random49': 270.0, 'random50': 295.0, 'random51': 263.0, 'random52': 295.0, 'random53': 300.0, 'random54': 247.0, 'random55': 259.0, 'random56': 284.0, 'random57': 303.0, 'random58': 271.0, 'random59': 263.0, 'random60': 267.0, 'random61': 275.0, 'random62': 259.0, 'random63': 280.0, 'cuccaro_adder': 146.0, 'deutsch_jozsa': 115.0, 'drapper_adder': 522.0, 'graph_state': 575.0, 'qft': 451.0, 'qnn': 1618.0, 'quantum_volume': 1075.0}}

  for (name, algo) in algos.items():
    my_results[name] = {}
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
  
  print(my_results)

  my_results = pd.DataFrame.from_dict(my_results, orient="index").T
  all_res = pd.concat([base_res, my_results], axis=1)
  print(all_res.to_csv(index=True))
  print(all_res)
