import json
import os
import pandas as pd
from contextlib import contextmanager
# from main import show_arch_comp_results, get_results_from_folder
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from utils.customtypes import Circuit
import torch
from time import time
from random import randint
from sampler.hardwaresampler import MultiTopologyHardwareSampler
from sampler.randomcircuit import RandomCircuit
from sampler.mixedcircuitsampler import MixedCircuitSampler
from qalloczero.alg.directalloc import DirectAllocator, DAConfig
from qalloczero.scripts.test_compare import validate, benchmark, compare_w_sota
from qalloczero.alg.ts import ModelConfigs
from utils.customtypes import Hardware
from collections import defaultdict


def _read_status_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError(f"{field} not found in /proc/self/status")


@contextmanager
def track_peak_memory(device=None):
    """Exact peak RAM (kernel-tracked) and peak VRAM (PyTorch) inside a with-block.

    Yields a dict that is filled in when the block exits:
        ram_peak            peak resident memory of this process
        ram_delta           peak minus RSS at block entry
        vram_peak           peak memory held by PyTorch tensors (CUDA only)
        vram_delta          peak minus allocation at block entry (CUDA only)
        vram_reserved_peak  peak memory held by PyTorch's cache (CUDA only)

    Not safe to nest: an inner block resets the counter the outer block relies on.
    """
    stats = {}
    # Tells the kernel to reset this process's peak-RSS counter (VmHWM) to the current RSS.
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    ram_start = _read_status_kb("VmRSS") * 1024
    _HAS_CUDA = torch.cuda.is_available()

    if _HAS_CUDA:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        vram_start = torch.cuda.memory_allocated(device)

    try:
        yield stats
    finally:
        ram_peak = _read_status_kb("VmHWM") * 1024
        stats["ram_peak"] = ram_peak
        stats["ram_delta"] = (ram_peak - ram_start)
        if _HAS_CUDA:
            torch.cuda.synchronize(device)
            vram_peak = torch.cuda.max_memory_allocated(device)
            stats["vram_peak"] = vram_peak
            stats["vram_delta"] = (vram_peak - vram_start)
            stats["vram_reserved_peak"] = torch.cuda.max_memory_reserved(device)


def _scalingWorker(hardware, n_circs, nq, d, allocator_seq, allocator_par):
    data = {'seq': defaultdict(list), 'par': defaultdict(list)}
    circ_sampler = RandomCircuit(num_lq=nq, num_slices=d)
    da_cfg = DAConfig()
    for i in range(n_circs):
        print(f"{i},", end='', flush=True)
        circ = circ_sampler.sample()
        with track_peak_memory() as mem_seq:
            t0 = time()
            cost_seq = allocator_seq.optimize(circ, cfg=da_cfg, hardware=hardware)[1]
            t1 = time()
        data['seq']['costs'].append(cost_seq/(circ.n_gates + 1))
        data['seq']['times'].append(t1-t0)
        data['seq']['ram_peak'].append(mem_seq['ram_peak'])
        data['seq']['ram_delta'].append(mem_seq['ram_delta'])
        data['seq']['vram_peak'].append(mem_seq['vram_peak'])
        data['seq']['vram_delta'].append(mem_seq['vram_delta'])
        data['seq']['vram_reserved_peak'].append(mem_seq['vram_reserved_peak'])
        with track_peak_memory() as mem_par:
            t2 = time()
            cost_par = allocator_par.optimize(circ, cfg=da_cfg, hardware=hardware)[1]
            t3 = time()
        data['par']['costs'].append(cost_par/(circ.n_gates + 1))
        data['par']['times'].append(t3-t2)
        data['par']['ram_peak'].append(mem_par['ram_peak'])
        data['par']['ram_delta'].append(mem_par['ram_delta'])
        data['par']['vram_peak'].append(mem_par['vram_peak'])
        data['par']['vram_delta'].append(mem_par['vram_delta'])
        data['par']['vram_reserved_peak'].append(mem_par['vram_reserved_peak'])
    print()
    return dict(data)


def _saveData(data, name):
    with open(name, 'w') as f:
        json.dump(data, f, indent=2)


def _loadData(name):
    with open(name, 'r') as f:
        data = json.load(f)
    return data


def scalingTest(compute: bool):
    depth_data = {}
    qubits_data = {}
    cores_data = {}
    n_circs=10
    name = 'da_v2_ft'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    allocator_seq = DirectAllocator.load(f'trained/{name}', device, checkpoint=-1).set_mode(DirectAllocator.Mode.Sequential)
    allocator_par = DirectAllocator.load(f'trained/{name}', device, checkpoint=-1).set_mode(DirectAllocator.Mode.Parallel)
    if compute:
        depths = [16*(2**i) for i in range(0,10)]
        n_qubits = [16*(2**i) for i in range(0,10)]
        n_cores = [2*(2**i) for i in range(1,7)]
        if not os.path.exists('data/depth_scaling.json'):
            for d in depths:
                print(f'Depth {d} from {depths}')
                nq=64
                n_cores=8
                hardware = Hardware(
                    core_capacities=torch.tensor([n_cores]*(nq//n_cores)),
                    core_connectivity=(torch.ones(n_cores,n_cores) - torch.eye(n_cores))
                )
                depth_data[d] = _scalingWorker(hardware, n_circs, nq, d, allocator_seq, allocator_par)
            _saveData(depth_data, 'data/depth_scaling.json')
        else:
            print('Skipping depth_scaling')
        if not os.path.exists('data/qubit_scaling.json'):
            for q in n_qubits:
                print(f'Qubits {q} from {n_qubits}')
                depth=64
                n_cores=8
                hardware = Hardware(
                    core_capacities=torch.tensor([n_cores]*(q//n_cores)),
                    core_connectivity=(torch.ones(n_cores,n_cores) - torch.eye(n_cores))
                )
                qubits_data[q] = _scalingWorker(hardware, n_circs, q, depth, allocator_seq, allocator_par)
            _saveData(qubits_data, 'data/qubit_scaling.json')
        else:
            print('Skipping qubit_scaling')
        if not os.path.exists('data/core_scaling.json'):
            for c in n_cores:
                print(f'Qubits {q} from {n_qubits}')
                depth=64
                nq=256
                hardware = Hardware(
                    core_capacities=torch.tensor([c]*(nq//c)),
                    core_connectivity=(torch.ones(c,c) - torch.eye(c))
                )
                cores_data[c] = _scalingWorker(hardware, n_circs, nq, depth, allocator_seq, allocator_par)
            _saveData(cores_data, 'data/core_scaling.json')
        else:
            print('Skipping core_scaling')
    depth_data = _loadData('data/depth_scaling.json')
    qubits_data = _loadData('data/qubit_scaling.json')
    cores_data = _loadData('data/core_scaling.json')
    # TODO


def topologyTest(compute: bool):
    if compute:
        n_circs = 10
        n_qubits = 64
        n_cores = 8
        n_slices = 32
        name = 'da_v2_ft'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        allocator_seq = DirectAllocator.load(f'trained/{name}', device, checkpoint=-1).set_mode(DirectAllocator.Mode.Sequential)
        allocator_par = DirectAllocator.load(f'trained/{name}', device, checkpoint=-1).set_mode(DirectAllocator.Mode.Parallel)
        hws = MultiTopologyHardwareSampler(n_qubits, range_ncores=[n_cores, n_cores])
        topology_names = ['all2all', 'linear', 'star', 'ring', 'grid']
        results = {}
        for tn in topology_names:
            hw = hws.sample()
            results[tn] = _scalingWorker(hw, n_circs, n_cores, n_slices, allocator_seq, allocator_par)  
        _saveData(results, 'data/topology_study.json')
    topology_data = _loadData('data/topology_data.json')
    # TODO

if __name__ == '__main__':
    # scalingTest(compute=True)
    topologyTest(compute=True)