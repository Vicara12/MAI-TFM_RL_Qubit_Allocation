import torch
from utils.timer import Timer
from utils.customtypes import Hardware, Circuit
from sampler.randomcircuit import RandomCircuit
from utils.allocutils import check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.directalloc import DirectAllocator
from russo.tests.hungarian import HQA


def validate():
  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 32
  n_circuits = 32
  core_caps = torch.tensor([4]*4, dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  algos = dict(
    hqa = HQA(lookahead=True, verbose=False),
    da_sequential = DirectAllocator.load("trained/da_v4", device="cpu").set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load("trained/da_v4", device="cpu").set_mode(DirectAllocator.Mode.Parallel),
    azero =               AlphaZero.load("trained/da_v4", device="cpu"),
    # da_azero = DirectAllocator.load("trained/azero", device="cuda"),
    # azero_azero =    AlphaZero.load("trained/azero", device="cpu"),
  )
  cfg = TSConfig(
    target_tree_size=512,
    noise=0.10,
    dirichlet_alpha=0.25,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.275,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  circuits = [sampler.sample() for _ in range(n_circuits)]

  for (name, algo) in algos.items():
    print(f"[*] Optimizing {name}")
    with Timer.get('t'):
      if isinstance(algo, DirectAllocator) or isinstance(algo, HQA):
        results = []
        for circ in circuits:
          results.append(algo.optimize(circ, hardware=hardware))
      elif isinstance(algo, AlphaZero):
        results = algo.optimize_mult(circuits, cfg, hardware=hardware)
      else:
        raise Exception("Unrecognized algorithm type")
    norm_res = torch.tensor([res[1]/circ.n_gates_norm for (res, circ) in zip(results,circuits)])
    norm_swaps = [
      count_swaps(swaps_from_alloc(res[0], n_cores))/circ.n_gates_norm for (res, circ) in zip(results,circuits)
    ]
    print(f" + t={Timer.get('t').time:.2f}s avg_cost={norm_res.mean().item():.4f} ({norm_res.std().item():.2f}) avg_swaps={sum(norm_swaps)/len(norm_swaps):.4f}")


def benchmark():
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
    # hqa = HQA(lookahead=True, verbose=False),
    da_sequential = DirectAllocator.load("trained/da_v4", device="cuda").set_mode(DirectAllocator.Mode.Sequential),
    da_parallel   = DirectAllocator.load("trained/da_v4", device="cuda").set_mode(DirectAllocator.Mode.Parallel),
    azero =               AlphaZero.load("trained/da_v4", device="cpu"),
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