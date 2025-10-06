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
