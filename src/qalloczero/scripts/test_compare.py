import torch
from utils.timer import Timer
from utils.customtypes import Hardware
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
  n_circuits = 16
  core_caps = torch.tensor([4]*4, dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  algos = dict(
    hqa = HQA(lookahead=True, verbose=False),
    da_trained = DirectAllocator.load("trained/da_v5", device="cuda"),
    # azero_trained =    AlphaZero.load("trained/da_v3", device="cpu"),
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
    norm_res = [res[1]/circ.n_gates_norm for (res, circ) in zip(results,circuits)]
    # norm_swaps = [
    #   count_swaps(swaps_from_alloc(res[0], n_cores))/circ.n_gates_norm for (res, circ) in zip(results,circuits)
    # ]
    norm_swaps = [1,1]
    print(f" + t={Timer.get('t').time:.2f}s avg_cost={sum(norm_res)/len(norm_res):.4f} avg_swaps={sum(norm_swaps)/len(norm_swaps):.4f}")