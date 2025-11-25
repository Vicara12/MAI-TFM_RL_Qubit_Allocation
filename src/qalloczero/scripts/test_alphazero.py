import torch
from utils.timer import Timer
from utils.customtypes import Hardware, Circuit
from utils.plotter import drawQubitAllocation
from utils.allocutils import sol_cost, check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap
from utils.other_utils import save_train_data
from sampler.randomcircuit import RandomCircuit
from sampler.hardwaresampler import HardwareSampler
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.ts_python import TSPythonEngine
from qalloczero.alg.ts_cpp import TSCppEngine
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.models.predmodel import PredictionModel


def testing_pred_model():
  core_connectivity = torch.tensor([
    [0,1],
    [1,0],
  ], dtype=torch.float)
  pred_model = PredictionModel(
    layers=[8,16,32]
  )
  pred_model.eval()
  qubits = torch.tensor([
    [1, 0],
    [2,-1],
    [2, 1],
  ])
  prev_core_allocs = torch.tensor([
    [0,1,1,1],
    [1,0,1,0],
    [2,2,2,2],
  ])
  current_core_allocs = torch.tensor([
    [0,2,1,2],
    [0,2,0,2],
    [2,2,2,2],
  ])
  core_capacities = torch.tensor([
    [ 1, 1],
    [ 0, 2],
    [ 2, 2],
  ], dtype=torch.float)
  circuit_emb = torch.randn(3,4,4)
  slice_adj_mat = torch.tensor([
    [[0,1,0,0],
     [1,0,0,0],
     [0,0,0,0],
     [0,0,0,0]],
    [[0,0,0,1],
     [0,0,0,0],
     [0,0,0,0],
     [1,0,0,0]],
    [[0,1,0,0],
     [1,0,0,0],
     [0,0,0,1],
     [0,0,1,0]],
  ])
  slice_gates = (((0,1),),((0,3),),((0,1),(2,3)))
  circ = Circuit(slice_gates=slice_gates, n_qubits=4)
  circuit_emb = circ.embedding
  
  probs_batched, vals_batched, _ = pred_model(
    qubits=qubits,
    prev_core_allocs=prev_core_allocs,
    current_core_allocs=current_core_allocs,
    core_capacities=core_capacities,
    core_connectivity=core_connectivity,
    circuit_emb=circuit_emb,
  )

  probs_single = []
  vals_single = []
  for i in range(len(slice_adj_mat)):
    p,v,l = pred_model(
      qubits=qubits[i].unsqueeze(0),
      prev_core_allocs=prev_core_allocs[i].unsqueeze(0),
      current_core_allocs=current_core_allocs[i].unsqueeze(0),
      core_capacities=core_capacities[i].unsqueeze(0),
      core_connectivity=core_connectivity,
      circuit_emb=circuit_emb[i].unsqueeze(0),
    )
    probs_single.append(p)
    vals_single.append(v)
  probs_single = torch.vstack(probs_single)
  vals_single = torch.vstack(vals_single)

  print(torch.equal(probs_batched, probs_single))
  print(torch.equal(vals_batched, vals_single))
  pass


def test_cpp_engine():
  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 1
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_connectivity = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  pred_model = PredictionModel(
    n_qubits=n_qubits,
    n_cores=core_connectivity.shape[0],
    number_emb_size=4,
    n_heads=4,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  circuit = sampler.sample()
  print(f"{circuit.slice_gates = }")
  encoder = None # CircuitEncoder(n_qubits=n_qubits, n_heads=4, n_layers=4)
  encoder.eval()
  params = dict(
    n_qubits=n_qubits,
    n_cores=n_cores,
    device="cpu",
  )
  cpp_engine = TSCppEngine(**params)
  py_engine = TSPythonEngine(**params)
  cpp_engine.load_model(pred_model)
  py_engine.load_model(pred_model)
  opt_params = dict(
    core_conns=core_connectivity,
    core_caps=core_caps,
    slice_adjm=circuit.adj_matrices,
    circuit_embs=encoder(circuit.adj_matrices.unsqueeze(0)).squeeze(0),
    alloc_steps=circuit.alloc_steps,
    cfg=TSConfig(target_tree_size=1024),
    ret_train_data=False,
    verbose=True,
  )
  torch.manual_seed(42)
  print("Starting C++ optimization:")
  with Timer.get('t1'):
    res_cpp = cpp_engine.optimize(**opt_params)
  cost = sol_cost(res_cpp[0], core_connectivity)
  print(f"t={Timer.get('t1').time:.2f}s cost={cost}")
  # print(f"t={Timer.get('t1').time:.2f}s cost={cost} {res_cpp = }")
  torch.manual_seed(42)
  print("Starting Python optimization:")
  with Timer.get('t2'):
    res_py = py_engine.optimize(**opt_params)
  cost = sol_cost(res_py[0], core_connectivity)
  print(f"t={Timer.get('t2').time:.2f}s cost={cost}")
  # print(f"t={Timer.get('t2').time:.2f}s cost={cost} {res_py = }")


def test_alphazero():
  test_run = True
  test_train = False

  # test_run = False
  # test_train = True

  test_parallel = False

  print("[*] TESTING ALPHA ZERO")
  

  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 8
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  hardware_sampler = HardwareSampler(max_nqubits=32, range_ncores=[2,8])
  # azero = AlphaZero.load("trained/da", device="cpu")
  azero = AlphaZero(
    device='cpu',
    backend=AlphaZero.Backend.Cpp,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  cfg = TSConfig(
    target_tree_size=64,
    noise=0.00,
    dirichlet_alpha=0.7,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.025,
    ucb_c2=500,
  )

  if test_run:
    cfg = TSConfig(
      target_tree_size=512,
      noise=0.2,
      dirichlet_alpha=1.0,
      discount_factor=0.0,
      action_sel_temp=0,
      ucb_c1=0.15,
      ucb_c2=500,
    )
    # azero = AlphaZero.load("trained/direct_allocator", device="cpu")
    # circuit = sampler.sample()
    circuit = Circuit(slice_gates=(((15, 14), (2, 11), (3, 9)), ((14, 10),), ((9, 10),), ((10, 14),), ((14, 1), (4, 13), (3, 12)), ((13, 5), (11, 3)), ((13, 6), (15, 10)), ((15, 5), (12, 6))), n_qubits=16)
    # orig: t=7.09s c=1.417 er=0.176 sw=9 (0.750)
    torch.manual_seed(42)
    with Timer.get('t'):
      allocs, cost, _, er = azero.optimize(circuit, cfg, hardware=hardware, verbose=False)
      # azero.optimize_mult([sampler.sample() for _ in range(10)], cfg, hardware=hardware)
    swaps = swaps_from_alloc(allocs, n_cores)
    n_swaps = count_swaps(swaps)
    check_sanity_swap(allocs, swaps)
    print(f"t={Timer.get('t').time:.2f}s c={cost/circuit.n_gates_norm:.3f} er={er:.3f} sw={n_swaps} ({n_swaps/circuit.n_gates_norm:.3f})")
    # drawQubitAllocation(allocs, core_caps, circuit.slice_gates, file_name="allocation.svg")
    # torch.manual_seed(42)
    # with Timer.get('t'):
    #   allocs, cost, _, _ = azero_loaded.optimize(circuit, cfg, verbose=True)
    # print(f"t={Timer.get('t').time:.2f}s c={cost}/{circuit.n_gates_norm} ({cost/circuit.n_gates_norm:.3f})\n{allocs}")
  
  if test_parallel:
    n_circuits = 1
    circuits = [sampler.sample() for _ in range(n_circuits)]
    # Series optimize
    torch.manual_seed(42)
    print("[*] Optimization in series")
    with Timer.get('t'):
      for i, circuit in enumerate(circuits):
        allocs, cost, _, _ = azero.optimize(circuit, cfg, hardware=hardware, verbose=False)
        check_sanity(allocs, circuit, hardware)
        print(f"[{i+1}/{n_circuits}] c={cost}/{circuit.n_gates_norm} ({cost/circuit.n_gates_norm:.3f})")
    print(f"Final t={Timer.get('t').time:.2f}s")
    
    # Parallel optimize
    torch.manual_seed(42)
    print("\n[*] Optimization in parallel")
    with Timer.get('t'):
      results = azero.optimize_mult(circuits, cfg, hardware=hardware)
    for i, res in enumerate(results):
      allocs, cost, _, _ = res
      check_sanity(allocs, circuits[i], hardware)
      print(f"[{i+1}/{n_circuits}] c={cost/circuits[i].n_gates_norm:.3f}")
    print(f"Final t={Timer.get('t').time:.2f}s")
  
  if test_train:
    cfg = TSConfig(
      target_tree_size=512,
      noise=1,
      dirichlet_alpha=0.25,
      discount_factor=0.0,
      action_sel_temp=0,
      ucb_c1=0.2,
      ucb_c2=500,
    )
    # azero = AlphaZero.load("trained/azero_finetune", device="cpu")
    azero = AlphaZero(
      device='cpu',
      backend=AlphaZero.Backend.Cpp,
    )
    save_dir = "trained/azero"
    try:
      sampler = RandomCircuit(num_lq=n_qubits, num_slices=4)
      train_cfg = AlphaZero.TrainConfig(
        train_iters=1_000,
        batch_size=16,
        n_data_augs=1,
        circ_sampler=sampler,
        hardware_sampler=hardware_sampler,
        noise_decrease_factor=0.975,
        lr=5e-5,
        ts_cfg=cfg,
        # print_grad_each=5,
        # detailed_grad=False,
      )
      train_data = azero.train(train_cfg, train_device='cuda')
      save_dir = azero.save(save_dir, overwrite=False)
      save_train_data(data=train_data, train_folder=save_dir)
    except KeyboardInterrupt:
      pass
    except Exception:
      azero.save(save_dir, overwrite=False)
      raise