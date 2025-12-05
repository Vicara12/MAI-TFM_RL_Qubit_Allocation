import os
import json
import torch
from typing import Optional
from random import randint
from sampler.hardwaresampler import HardwareSampler
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.directalloc import DirectAllocator
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.alg.ts import ModelConfigs, TSConfig
from utils.other_utils import save_train_data
from utils.customtypes import Circuit, Hardware
from utils.timer import Timer
from utils.allocutils import count_swaps, swaps_from_alloc



def arch_comparison(architectures: dict[str, list[int]], subfolder: str):
  validation_hardware = Hardware(
    core_capacities=torch.tensor([4]*4),
    core_connectivity=(torch.ones(4,4) - torch.eye(4))
  )
  hardware_sampler = HardwareSampler(max_nqubits=40, range_ncores=[2,8])

  for i, (name, arch) in enumerate(architectures.items()):
    print(f"\n ======== Training architecture {name} ({i+1}/{len(architectures)}): {arch} ========")
    allocator = DirectAllocator(
      device='cuda',
      model_cfg=ModelConfigs(layers=arch),
    )
    train_cfg = DirectAllocator.TrainConfig(
      train_iters=150,
      batch_size=8 ,
      validation_size=20,
      initial_noise=0.4,
      noise_decrease_factor=0.95,
      circ_sampler=RandomCircuit(num_lq=16, num_slices=16),
      lr=5e-5,
      invalid_move_penalty=0.3,
      hardware_sampler=hardware_sampler,
    )
    save_folder = f"trained/{subfolder}/da_{name}"
    train_data = allocator.train(train_cfg, validation_hardware=validation_hardware)
    save_folder = allocator.save(save_folder, overwrite=False)
    save_train_data(data=train_data, train_folder=save_folder)


def get_results_from_folder(folder):
  results = dict()
  for d in os.listdir(folder):
    if os.path.isdir(os.path.join(folder,d)):
      with open(os.path.join(folder,d,'train_data.json'), 'r') as f:
        results[d] = json.load(f)
  return results


def show_arch_comp_results(folder):
  results = get_results_from_folder(folder)
  cost = {}
  for k in sorted(results.keys()):
    n = 10
    cost[k] = sum(results[k]['validation_cost'][-n:])/n
    print(f"{k}: {cost[k]:.3f}")
  return cost


def architecture_H_comparison():
  arch_comparison({
    8:   [8,  8,  8,  8,   8,   8],
    16:  [8, 16, 16, 16,  16,  16],
    32:  [8, 16, 32, 32,  32,  32],
    64:  [8, 16, 32, 64,  64,  64],
    128: [8, 16, 32, 64, 128, 128],
    256: [8, 16, 32, 64, 128, 256],
  }, "architecture")
  show_arch_comp_results('trained/architecture')


def architecture_shape_comparison():
  arch_comparison({
    'direct': [      8,         16,             32],
    'double': [   8, 8,     16, 16,         32, 32],
    'triple': [8, 8, 8, 16, 16, 16,     32, 32, 32],
    'end':    [   8, 8,     16, 16, 32, 32, 32, 32]
  }, "shape")
  show_arch_comp_results('trained/shape')


def train_model_da(allocator, name: str):
  validation_hardware = Hardware(
    core_capacities=torch.tensor([4]*4),
    core_connectivity=(torch.ones(4,4) - torch.eye(4))
  )
  val_sampler = RandomCircuit(num_lq=16, num_slices=32)
  train_cfg = DirectAllocator.TrainConfig(
    train_iters=4_000,
    batch_size=1,
    group_size=32,
    validate_each=25,
    validation_hardware=validation_hardware,
    validation_circuits=[val_sampler.sample() for _ in range(32)],
    store_path=f"trained/{name}",
    initial_noise=0.2,
    noise_decrease_factor=0.9975,
    min_noise=0.0,
    circ_sampler=RandomCircuit(num_lq=24, num_slices=lambda: randint(8,16), reflow=0.5),
    lr=5e-5,
    inv_mov_penalization=0.3,
    mask_invalid=False,
    hardware_sampler=HardwareSampler(max_nqubits=24, range_ncores=[2,8]),
  )
  allocator.train(train_cfg)


def optimize(allocator, hardware, circuit, cfg: Optional[TSConfig] = None):
  with Timer.get('t'):
    if isinstance(allocator, DirectAllocator):
      result = allocator.optimize(circuit, hardware=hardware)
    elif isinstance(allocator, AlphaZero):
      result = allocator.optimize(circuit, cfg, hardware=hardware)
    else:
      raise Exception("Unrecognized algorithm type")
  norm_cost = result[1]/circuit.n_gates_norm
  norm_swaps = count_swaps(swaps_from_alloc(result[0], hardware.n_cores))/circuit.n_gates_norm
  print(f" + t={Timer.get('t').time:.2f}s cost={result[1]} norm_cost={norm_cost} swaps={norm_swaps}")


def train_azero(azero: AlphaZero, name: str):
  save_dir = f"trained/{name}"
  cfg = TSConfig(
    target_tree_size=512,
    noise=1,
    dirichlet_alpha=1.0,
    discount_factor=0.0,
    action_sel_temp=1,
    ucb_c1=0.125,
    ucb_c2=500,
  )
  # num_lq is not important as it will be derived from sampled hardware
  circuit_sampler = RandomCircuit(num_lq=4, num_slices=16)
  hardware_sampler = HardwareSampler(max_nqubits=24, range_ncores=[4,8])
  train_cfg = AlphaZero.TrainConfig(
    train_iters=1_000,
    batch_size=16,
    n_data_augs=1,
    circ_sampler=circuit_sampler,
    hardware_sampler=hardware_sampler,
    noise_decrease_factor=0.975,
    lr=5e-3,
    ts_cfg=cfg,
  )
  try:
    train_data = azero.train(train_cfg, train_device='cuda')
    save_dir = azero.save(save_dir, overwrite=False)
    save_train_data(data=train_data, train_folder=save_dir)
  except KeyboardInterrupt:
    pass
  except Exception:
    azero.save(save_dir, overwrite=False)
    raise


def benchmark(allocator, cfg: Optional[TSConfig] = None):
  circuits = [
    "qft", # Exact
    "quantum_volume",
    "graph_state", # Exact
    "drapper_adder",
    "cuccaro_adder", # A bit over
    "qnn",
    "deutsch_jozsa", # Exact
  ]
  # A2A configuration
  hardware = Hardware(
    core_capacities=torch.tensor([10]*10),
    core_connectivity=(torch.ones(size=(10,10)) - torch.eye(10)),
  )

  for name in circuits:
    # circuit50 = Circuit.from_qasm(f'circuits/{name}_50.qasm', 50)
    circuit100 = Circuit.from_qasm(f'circuits/{name}_100.qasm', 100)
    print(f"[*] Optimizing {name}")
    optimize(allocator, hardware, circuit100, cfg)


def benchmark_da(checkpoint: str):
  benchmark(allocator=DirectAllocator.load(checkpoint, device="cuda"))


def benchmark_azero(checkpoint: str):
  cfg = TSConfig(
    target_tree_size=512,
    noise=0.10,
    dirichlet_alpha=0.25,
    discount_factor=0.0,
    action_sel_temp=0,
    ucb_c1=0.275,
  )
  benchmark(allocator=AlphaZero.load(checkpoint, device="cpu"), cfg=cfg)


if __name__ == "__main__":
  ''' Test different embedding (H) sizes '''
  # architecture_H_comparison()

  ''' Test different layer distributions and depths '''
  # architecture_shape_comparison()

  ''' Train the base models with direct allocation '''
  allocator = DirectAllocator(
    device='cuda',
    model_cfg=ModelConfigs(embed_size=32, num_heads=2, num_layers=2),
    mode=DirectAllocator.Mode.Sequential,
  )
  train_model_da(allocator, name="da")

  ''' Refine a direct allocator model '''
  # allocator = DirectAllocator.load('trained/da_v10')
  # train_model_da(allocator, name="da_v10_ft")

  ''' Train the base models with qalloczero '''
  # train_azero(AlphaZero(model_cfg=ModelConfigs(layers=[16,32])), name="az")

  ''' Benchmark with real circuits using Direct Allocation '''
  # benchmark_da("trained/da_v3")

  ''' Benchmark with real circuits using AlphaZero '''
  # benchmark_azero("trained/da_v3")
