import torch
from utils.timer import Timer
from utils.customtypes import Hardware
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.ts import TSConfig
from qalloczero.alg.alphazero import AlphaZero


torch.manual_seed(42)
n_qubits = 16
n_slices = 32
core_caps = torch.tensor([4,4,4,4], dtype=torch.int)
n_cores = core_caps.shape[0]
core_conn = torch.ones((n_cores,n_cores)) - torch.eye(n_cores)
hardware = Hardware(core_capacities=core_caps, core_connectivity=core_conn)
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


cfg = TSConfig(
  target_tree_size=512,
  noise=0.0,
  dirichlet_alpha=1.0,
  discount_factor=0.0,
  action_sel_temp=0,
  ucb_c1=0.15,
  ucb_c2=500,
)
circuit = sampler.sample()
torch.manual_seed(42)
with Timer.get('t'):
  cost, er, tdata = azero._optimize_mult_train([circuit], hardware=hardware, ts_cfg=cfg)[0]
print(f"t={Timer.get('t').time:.2f}s c={cost*(circuit.n_gates_norm+1):.3f} ({cost}) er={er:.3f}")

alloc_steps = circuit.alloc_steps
for i in range(circuit.n_steps):
  print(f"Step {i+1}: {tdata.value[i]*(alloc_steps[i,3] + 1)}")