import torch
from utils.allocutils import swaps_from_alloc, check_sanity_swap

if __name__ == '__main__':
  tests = dict(
    simple_cycle = (2,[[0,1],
                       [1,0]]),
    simple_square = (4, [[0,1,2,3],
                         [1,2,3,0]]),
    double_square = (4, [[0,1,2,3,0,3,2,1],
                         [1,2,3,0,3,2,1,0]])
  )

  for (name, (n_cores, allocs)) in tests.items():
    print(f"Test: {name}")
    t_allocs = torch.tensor(allocs)
    print(f"{'\n'.join([str(a) for a in allocs])}")
    swaps = swaps_from_alloc(t_allocs, n_cores)
    print(f"{'\n'.join([f'Slices {i} - {i+1}: ' + ', '.join([str(s) for s in swaps_i]) for i, swaps_i in enumerate(swaps)])}\n")
    check_sanity_swap(t_allocs, swaps)