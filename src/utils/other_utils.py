import json
import os



def save_train_data(data: dict[str, list], train_folder: str):
  if not os.path.isdir(train_folder):
    raise Exception(f"Provided folder does not exist: {train_folder}")
  with open(os.path.join(train_folder, "train_data.json"), "w") as f:
      json.dump(data, f, indent=2)