import os
import json
from sampler.hardwaresampler import HardwareSampler
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.directalloc import DirectAllocator
from qalloczero.alg.ts import ModelConfigs
from utils.other_utils import save_train_data



def arch_comparison(architectures: dict[str, list[int]], subfolder: str):
  hardware_sampler = HardwareSampler(max_nqubits=40, range_ncores=[2,8])

  for i, (name, arch) in enumerate(architectures.items()):
    print(f"\n ======== Training architecture {name} ({i+1}/{len(architectures)}): {arch} ========")
    allocator = DirectAllocator(
      hardware_sampler.sample(), # Random default hardware
      device='cuda',
      model_cfg=ModelConfigs(layers=arch),
    )
    train_cfg = DirectAllocator.TrainConfig(
      train_iters=200,
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
    train_data = allocator.train(train_cfg)
    save_folder = allocator.save(save_folder, overwrite=False)
    save_train_data(data=train_data, train_folder=save_folder)


def show_arch_comp_results(folder):
  results = dict()
  for d in os.listdir(folder):
    if os.path.isdir(os.path.join(folder,d)):
      with open(os.path.join(folder,d,'train_data.json'), 'r') as f:
        results[d] = json.load(f)
  for k,v in results.items():
    n = 15
    val_cost = sum(v['validation_cost'][-n:])/n
    print(f"{k}: {val_cost:.3f}")


def architecture_H_comparison():
  arch_comparison({
    8: [8,8],
    16: [8,8,16,16],
    32: [8,8,16,16,32,32],
    64: [8,8,16,16,32,43,64,64],
    128: [8,8,16,16,32,32,64,64,128,128],
    256: [8,8,16,16,32,32,64,64,128,128,256,256],
  }, "architecture")
  show_arch_comp_results('trained/architecture')


def architecture_shape_comparison():
  arch_comparison({
    'direct': [8,16,32,64,128],
    'end': [8,16,32,64,128,128,128],
    'double': [8,8,16,16,32,32,64,64,128,128],
    'middle': [8,16,32,32,32,64,128],
    'beg': [8,8,8,16,32,64,128],
  }, "shape")
  show_arch_comp_results('trained/shape')


def train_model_da():
  hardware_sampler = HardwareSampler(max_nqubits=40, range_ncores=[2,8])
  allocator = DirectAllocator(
    hardware_sampler.sample(), # Default hardware is not important
    device='cuda',
    model_cfg=ModelConfigs(layers=[8,8,16,16,32,32,64,63,128,128]),
  )
  save_folder = "trained/da"
  try:
    train_cfg = DirectAllocator.TrainConfig(
      train_iters=1_000,
      batch_size=8 ,
      validation_size=20,
      initial_noise=0.4,
      noise_decrease_factor=0.95,
      circ_sampler=RandomCircuit(num_lq=50, num_slices=16),
      lr=5e-5,
      invalid_move_penalty=0.3,
      hardware_sampler=hardware_sampler,
    )
    train_data = allocator.train(train_cfg)
    save_folder = allocator.save(save_folder, overwrite=False)
    save_train_data(data=train_data, train_folder=save_folder)
  except KeyboardInterrupt:
    pass
  except Exception:
    allocator.save(save_folder, overwrite=False)
    raise


if __name__ == "__main__":
  ''' Test different embedding (H) sizes '''
  # architecture_H_comparison()

  ''' Test different layer distributions and depths '''
  # architecture_shape_comparison()

  ''' Train the base model with direct allocation '''
  train_model_da()