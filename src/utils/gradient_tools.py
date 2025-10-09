import torch



def print_graph(op, level=0):
  if level >= 10:
    print("  "*level, "Max depth...")
    return
  if op is None:
    print("  "*level, "Leaf:", op)
    return
  print("  "*level, type(op).__name__)
  for f in op.next_functions:
    print_graph(f[0], level+1)


def print_grad(model):
  for name, param in model.named_parameters():
    if param.grad is None:
      print(f"No grad: \t{name}")
    else:
      print(f"mean={param.grad.mean().item():.8f} \tabsmax={param.grad.abs().max().item():.8f}: \t{name}")


def grad_stats(model):
  max_grads = {}
  for name, param in model.named_parameters():
    if param.grad is not None:
      max_grads[name] = param.grad.abs().max().item()
  if not max_grads:
    return 0, 0, 0
  vals = torch.tensor(list(max_grads.values()))
  mean_grad = vals.mean().item()
  max_grad = max(max_grads.items(), key=lambda x: x[1])
  min_grad = min(max_grads.items(), key=lambda x: x[1])
  return mean_grad, max_grad, min_grad


def print_grad_stats(model: torch.nn.Module, name: str):
  mean_grad, max_grad, min_grad = grad_stats(model)
  print(f"+ Grad stats for {name}:\n    - mean={mean_grad:.8f}\n    - min={min_grad}\n    - max={max_grad}")