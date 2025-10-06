import torch
from utils.timer import Timer
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation
from utils.allocutils import check_sanity
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.ts import ModelConfigs
from qalloczero.alg.directalloc import DirectAllocator, DAConfig


def test_direct_alloc():
  # test_run = True
  # test_train = False

  test_run = False
  test_train = True

  test_parallel = False

  print("[*] TESTING DIRECT ALLOCATION")

  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 32
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  # allocator = DirectAllocator.load("trained/direct_allocator_v2", device="cuda")
  allocator = DirectAllocator(
    hardware,
    device='cuda',
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  cfg = DAConfig(
    noise=0.0,
    mask_invalid=True,
    greedy=True,
  )

  if test_run:
    circuit = sampler.sample()
    torch.manual_seed(42)
    with Timer.get('t'):
      allocs, cost, = allocator.optimize(circuit, cfg)
    print(f"t={Timer.get('t').time:.2f}s c={cost/circuit.n_gates_norm:.3f}\n{allocs}")
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
    try:
      train_cfg = DirectAllocator.TrainConfig(
        train_iters=1_000,
        batch_size=20,
        validation_size=10,
        initial_noise=0.3,
        noise_decrease_factor=0.95,
        sampler=RandomCircuit(num_lq=n_qubits, num_slices=4),
        lr=1e-4,
        invalid_move_penalty=0.3,
        repl_significance=0.05,
        print_grad_each=10,
      )
      allocator.train(train_cfg)
      allocator.save("trained/direct_allocator", overwrite=False)
    except KeyboardInterrupt:
      pass
    except Exception:
      allocator.save("trained/direct_allocator", overwrite=False)
      raise