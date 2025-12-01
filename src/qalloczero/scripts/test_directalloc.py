import torch
from random import randint
from utils.timer import Timer
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation
from utils.allocutils import check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap
from utils.other_utils import save_train_data
from sampler.randomcircuit import RandomCircuit
from sampler.hardwaresampler import HardwareSampler
from qalloczero.alg.ts import ModelConfigs
from qalloczero.alg.directalloc import DirectAllocator, DAConfig
from russo.tests.hungarian import HQA


def test_direct_alloc():
  test_run = True
  test_train = False

  # test_run = False
  # test_train = True

  test_parallel = False

  print("[*] TESTING DIRECT ALLOCATION")

  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 32
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  hardware_sampler = HardwareSampler(max_nqubits=20, range_ncores=[2,8])

  # allocator = DirectAllocator.load("trained/direct_allocator", device="cuda")
  # allocator = DirectAllocator(
  #   device='cuda',
  # )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  cfg = DAConfig(
    noise=0.0,
    mask_invalid=True,
    greedy=True,
  )

  if test_run:
    allocator = DirectAllocator.load("trained/da_v6", device="cpu").set_mode(DirectAllocator.Mode.Parallel)
    circuit = sampler.sample()
    circuit.alloc_slices
    torch.manual_seed(42)
    with Timer.get('t'):
      allocs, cost, = allocator.optimize(circuit, hardware=hardware, cfg=cfg)
    swaps = swaps_from_alloc(allocs, n_cores)
    n_swaps = count_swaps(swaps)
    check_sanity_swap(allocs, swaps)
    print(f"t={Timer.get('t').time:.2f}s c={cost/circuit.n_gates_norm:.3f} sw={n_swaps} ({n_swaps/circuit.n_gates_norm:.3f})\n{allocs}")
    check_sanity(allocs, circuit, hardware)
    drawQubitAllocation(allocs, core_caps, circuit.slice_gates, file_name="allocation.svg")
  
  if test_parallel:
    n_circuits = 1
    circuits = [sampler.sample() for _ in range(n_circuits)]
    # Series optimize
    torch.manual_seed(42)
    print("[*] Optimization in series")
    with Timer.get('t'):
      for i, circuit in enumerate(circuits):
        allocs, cost, _, _ = allocator.optimize(circuit, cfg, verbose=False)
        check_sanity(allocs, circuit, hardware)
        print(f"[{i+1}/{n_circuits}] c={cost/circuit.n_gates_norm:.3f}")
    print(f"Final t={Timer.get('t').time:.2f}s")
    
    # Parallel optimize
    torch.manual_seed(42)
    print("\n[*] Optimization in parallel")
    with Timer.get('t'):
      results = allocator.optimize_mult(circuits, cfg)
    for i, res in enumerate(results):
      allocs, cost, _, _ = res
      check_sanity(allocs, circuits[i], hardware)
      print(f"[{i+1}/{n_circuits}] c={cost/circuits[i].n_gates_norm:.3f}")
    print(f"Final t={Timer.get('t').time:.2f}s")
  
  if test_train:
    allocator = DirectAllocator(
      device='cpu',
      model_cfg=ModelConfigs(layers=[8,8,16,16,32,32,64,64,64]),
      mode=DirectAllocator.Mode.Parallel,
    )
    # allocator = DirectAllocator.load("trained/direct_allocator", device="cuda")
    train_cfg = DirectAllocator.TrainConfig(
      train_iters=1_000,
      batch_size=4,
      group_size=4,
      validate_each=25,
      validation_hardware=hardware,
      validation_circuits=[sampler.sample() for _ in range(64)],
      store_path=f"trained/test",
      initial_noise=0.8,
      noise_decrease_factor=0.99,
      circ_sampler=RandomCircuit(num_lq=16, num_slices=lambda: randint(8,16)),
      lr=5e-4,
      hardware_sampler=HardwareSampler(max_nqubits=16, range_ncores=[2,8]),
    )

    allocator.train(train_cfg)