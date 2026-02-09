import json
import os



def save_train_data(data: dict[str, list], train_folder: str):
  if not os.path.isdir(train_folder):
    raise Exception(f"Provided folder does not exist: {train_folder}")
  with open(os.path.join(train_folder, "train_data.json"), "w") as f:
      json.dump(data, f, indent=2)
      

def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)