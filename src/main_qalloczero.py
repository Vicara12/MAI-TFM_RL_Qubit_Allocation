import torch
from utils.timer import Timer
from utils.allocutils import solutionCost
from sampler.randomcircuit import RandomCircuit
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.predmodel import PredictionModel
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.ts_cpp import TSCppEngine
from qalloczero.alg.ts_python import TSPythonEngine




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
    n_qubits=4,
    n_cores=2,
    number_emb_size=4,
    n_heads=4,
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
  cost = solutionCost(res_cpp[0], core_connectivity)
  print(f"t={Timer.get('t1').time:.2f}s cost={cost}")
  # print(f"t={Timer.get('t1').time:.2f}s cost={cost} {res_cpp = }")
  torch.manual_seed(42)
  print("Starting Python optimization:")
  with Timer.get('t2'):
    res_py = py_engine.optimize(**opt_params)
  cost = solutionCost(res_py[0], core_connectivity)
  print(f"t={Timer.get('t2').time:.2f}s cost={cost}")
  # print(f"t={Timer.get('t2').time:.2f}s cost={cost} {res_py = }")


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
    costs.append(solutionCost(res_py[0], core_connectivity))
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


def test_alphazero():
  test_run = True
  test_train = False

  torch.manual_seed(42)
  n_qubits = 16
  n_slices = 8
  core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
  n_cores = core_caps.shape[0]
  core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
  azero = AlphaZero(
    hardware,
    device='cuda',
    backend=AlphaZero.Backend.Python,
  )
  sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
  cfg=TSConfig(target_tree_size=32)

  if test_run:
    azero.save("checkpoint", overwrite=True)
    azero_loaded = AlphaZero.load("checkpoint", device="cuda")
    circuit = sampler.sample()
    torch.manual_seed(42)
    with Timer.get('t'):
      allocs, cost, _, _ = azero.optimize(circuit, cfg, verbose=True)
    print(f"t={Timer.get('t').time:.2f}s c={cost}/{circuit.n_gates} ({cost/circuit.n_gates:.3f})\n{allocs}")
    drawQubitAllocation(allocs, core_caps, circuit.slice_gates, file_name="allocation.svg")
    torch.manual_seed(42)
    with Timer.get('t'):
      allocs, cost, _, _ = azero_loaded.optimize(circuit, cfg, verbose=True)
    print(f"t={Timer.get('t').time:.2f}s c={cost}/{circuit.n_gates} ({cost/circuit.n_gates:.3f})\n{allocs}")
    drawQubitAllocation(allocs, core_caps, circuit.slice_gates, file_name="allocation.svg")
  
  if test_train:
    train_cfg = AlphaZero.TrainConfig(
      train_iters=100,
      batch_size=3,
      sampler=sampler,
      lr=0.01,
      pol_loss_w=0.6,
      ts_cfg=cfg,
    )
    azero.train(train_cfg)



if __name__ == "__main__":
  # testing_circuit_enc()
  # testing_pred_model()
  # test_cpp_engine()
  # grid_search()
  test_alphazero()