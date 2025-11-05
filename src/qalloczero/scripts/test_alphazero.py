import torch
from utils.timer import Timer
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation
from utils.allocutils import sol_cost, check_sanity, swaps_from_alloc, count_swaps, check_sanity_swap
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.ts_python import TSPythonEngine
from qalloczero.alg.ts_cpp import TSCppEngine
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.predmodel import PredictionModel


def testing_circuit_enc():
  sampler = RandomCircuit(num_lq=4, num_slices=3)
  cs = [sampler.sample().adj_matrices.unsqueeze(0) for _ in range(2)]
  encoder = CircuitEncoder(n_qubits=4, n_heads=4, n_layers=4)
  encoder.eval()
  embs = [encoder(m) for m in cs]
  embs_b = encoder(torch.vstack(cs))
  print(torch.equal(torch.vstack(embs), embs_b))


def testing_pred_model():
  core_connectivity = torch.tensor([
    [0,1],
    [1,0],
  ], dtype=torch.float)
  pred_model = PredictionModel(
    # n_qubits=4,
    # n_cores=2,
    # number_emb_size=4,
    # n_heads=4,
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
     [0,0,1,0],
     [0,0,0,1]],
    [[0,0,0,1],
     [0,1,0,0],
     [0,0,1,0],
     [1,0,0,0]],
    [[0,1,0,0],
     [1,0,0,0],
     [0,0,0,1],
     [0,0,1,0]],
  ])
  circ_enc = CircuitEncoder(n_qubits=4, n_heads=0, n_layers=0)
  circuit_emb = circ_enc(slice_adj_mat.unsqueeze(0))[0]
  
  probs_batched, vals_batched = pred_model(
    qubits=qubits,
    prev_core_allocs=prev_core_allocs,
    current_core_allocs=current_core_allocs,
    core_capacities=core_capacities,
    core_connectivity=core_connectivity,
    circuit_emb=circuit_emb,
    slice_adj_mat=slice_adj_mat,
  )

  probs_single = []
  vals_single = []
  for i in range(len(slice_adj_mat)):
    p,v = pred_model(
      qubits=qubits[i].unsqueeze(0),
      prev_core_allocs=prev_core_allocs[i].unsqueeze(0),
      current_core_allocs=current_core_allocs[i].unsqueeze(0),
      core_capacities=core_capacities[i].unsqueeze(0),
      core_connectivity=core_connectivity,
      circuit_emb=circuit_emb[i].unsqueeze(0),
      slice_adj_mat=slice_adj_mat[i].unsqueeze(0),
    )
    probs_single.append(p)
    vals_single.append(v)
  probs_single = torch.vstack(probs_single)
  vals_single = torch.vstack(vals_single)

  print(torch.equal(probs_batched, probs_single))
  print(torch.equal(vals_batched, vals_single))


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
  encoder = CircuitEncoder(n_qubits=n_qubits, n_heads=4, n_layers=4)
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
  # test_run = True
  # test_train = False

  test_run = False
  test_train = True

  test_parallel = False

  print("[*] TESTING ALPHA ZERO")
  

  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 32
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  # azero = AlphaZero.load("trained/direct_allocator", device="cpu")
  # azero = AlphaZero(
  #   hardware,
  #   device='cpu',
  #   backend=AlphaZero.Backend.Cpp,
  # )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  cfg = TSConfig(
    target_tree_size=1024,
    noise=0.00,
    dirichlet_alpha=0.7,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.025,
    ucb_c2=500,
  )

  if test_run:
    cfg = TSConfig(
      target_tree_size=2048,
      noise=0.70,
      dirichlet_alpha=0.0,
      discount_factor=0.0,
      action_sel_temp=0,
      ucb_c1=0.05,
      ucb_c2=500,
    )
    azero = AlphaZero.load("trained/direct_allocator", device="cpu")
    circuit = sampler.sample()
    torch.manual_seed(42)
    with Timer.get('t'):
      allocs, cost, _, er = azero.optimize(circuit, cfg, verbose=True)
    swaps = swaps_from_alloc(allocs, n_cores)
    n_swaps = count_swaps(swaps)
    check_sanity_swap(allocs, swaps)
    print(f"t={Timer.get('t').time:.2f}s c={cost/circuit.n_gates_norm:.3f} er={er:.3f} sw={n_swaps} ({n_swaps/circuit.n_gates_norm:.3f})")
    drawQubitAllocation(allocs, core_caps, circuit.slice_gates, file_name="allocation.svg")
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
        allocs, cost, _, _ = azero.optimize(circuit, cfg, verbose=False)
        check_sanity(allocs, circuit, hardware)
        print(f"[{i+1}/{n_circuits}] c={cost}/{circuit.n_gates_norm} ({cost/circuit.n_gates_norm:.3f})")
    print(f"Final t={Timer.get('t').time:.2f}s")
    
    # Parallel optimize
    torch.manual_seed(42)
    print("\n[*] Optimization in parallel")
    with Timer.get('t'):
      results = azero.optimize_mult(circuits, cfg)
    for i, res in enumerate(results):
      allocs, cost, _, _ = res
      check_sanity(allocs, circuits[i], hardware)
      print(f"[{i+1}/{n_circuits}] c={cost/circuits[i].n_gates_norm:.3f}")
    print(f"Final t={Timer.get('t').time:.2f}s")
  
  if test_train:
    cfg = TSConfig(
      target_tree_size=256,
      noise=1,
      dirichlet_alpha=0.0,
      discount_factor=0.0,
      action_sel_temp=0,
      ucb_c1=0.005,
      ucb_c2=500,
    )
    # azero = AlphaZero.load("trained/azero_finetune", device="cpu")
    azero = AlphaZero(
      hardware,
      device='cpu',
      backend=AlphaZero.Backend.Cpp,
    )
    try:
      sampler = RandomCircuit(num_lq=n_qubits, num_slices=4)
      train_cfg = AlphaZero.TrainConfig(
        train_iters=2_000,
        batch_size=16,
        n_data_augs=1,
        sampler=sampler,
        noise_decrease_factor=0.975,
        lr=1e-5,
        ts_cfg=cfg,
        # print_grad_each=5,
        # detailed_grad=False,
      )
      azero.train(train_cfg, train_device='cuda')
      azero.save("trained/azero", overwrite=False)
    except KeyboardInterrupt:
      pass
    except Exception:
      azero.save("trained/azero", overwrite=False)
      raise