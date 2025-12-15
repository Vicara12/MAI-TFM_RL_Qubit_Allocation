import objgraph
import gc
import torch
import gc
import sys
from collections import defaultdict

import psutil
import os


def get_ram_usage():
  process = psutil.Process(os.getpid())
  
  # Get memory usage in bytes and convert to Megabytes (MB)
  # rss = Resident Set Size (physical memory currently used)
  mem_bytes = process.memory_info().rss 
  return mem_bytes / (1024 * 1024 * 1024)


def print_ram_usage():  
  print(f"Current RAM Usage: {get_ram_usage():.6f} GB")

def print_memory_top_10():
  # Dictionary to store total size per type
  type_sizes = defaultdict(int)
  type_counts = defaultdict(int)

  # iterate over all objects tracked by the Garbage Collector
  for obj in gc.get_objects():
    # Get the type name
    obj_type = type(obj).__name__
    
    # Special handling for Tensors to get accurate GPU/CPU memory size
    if torch.is_tensor(obj):
      # numel() * element_size() gives bytes
      size = obj.numel() * obj.element_size()
    else:
      # Standard Python object size (shallow)
      size = sys.getsizeof(obj)

    type_sizes[obj_type] += size
    type_counts[obj_type] += 1


  # Sort by total size (descending)
  sorted_stats = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)[:10]

  print(f"{'Type':<30} | {'Count':<10} | {'Total Size (MB)':<15}")
  print("-" * 65)
  
  for obj_type, total_size in sorted_stats:
    count = type_counts[obj_type]
    size_mb = total_size / (1024 * 1024)  # Convert bytes to MB
    print(f"{obj_type:<30} | {count:<10} | {size_mb:<15.2f}")


def obj_count_top():
  objgraph.show_growth(limit=10)
  # CHECK FOR LEAKING TENSORS
  # If this number grows steadily every batch, you have a leak.
  print(f"Active Tensors: {len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])}")



def all_top():
  print()
  print_ram_usage()
  print_memory_top_10()
  print("--- Memory Top 10 ---")
  obj_count_top()
