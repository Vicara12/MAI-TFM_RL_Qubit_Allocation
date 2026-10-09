import argparse
import json
import os
import random
import traceback
import pandas as pd
from contextlib import contextmanager
# from main import show_arch_comp_results, get_results_from_folder
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from utils.customtypes import Circuit
import torch
import torch.multiprocessing as mp
from time import time
from sampler.hardwaresampler import MultiTopologyHardwareSampler
from sampler.randomcircuit import RandomCircuit
from sampler.mixedcircuitsampler import MixedCircuitSampler
from qalloczero.alg.directalloc import DirectAllocator, DAConfig
from qalloczero.alg.hqa import HQA
from qalloczero.scripts.test_compare import validate, benchmark, compare_w_sota
from qalloczero.alg.ts import ModelConfigs
from utils.customtypes import Hardware
from utils.allocutils import check_sanity
from collections import defaultdict
from functools import lru_cache


MODEL_NAME = 'da_v2_ft'
DATA_DIR = 'data'
PARTIAL_DIR = os.path.join(DATA_DIR, 'partial')
METRICS = ['costs', 'times', 'ram_peak', 'ram_delta', 'vram_peak', 'vram_delta', 'vram_reserved_peak']
TOPOLOGIES = {
    'all2all': MultiTopologyHardwareSampler._coreConAllToAll,
    'linear':  MultiTopologyHardwareSampler._coreConLinear,
    'star':    MultiTopologyHardwareSampler._coreConStar,
    'ring':    MultiTopologyHardwareSampler._coreConRing,
    'grid':    MultiTopologyHardwareSampler._coreConGrid,
}
STUDY_FILES = {
    'depth':    'depth_scaling.json',
    'qubits':   'qubit_scaling.json',
    'cores':    'core_scaling.json',
    'topology': 'topology_study.json',
    'bench50':  'benchmark_50.json',
    'bench100': 'benchmark_100.json',
}
# Allocators evaluated in each study: DirectAllocator sequential/parallel and the HQA baseline
STUDY_MODES = {
    'depth':    ['seq', 'par'],
    'qubits':   ['seq', 'par'],
    'cores':    ['seq', 'par'],
    'topology': ['seq', 'par', 'hqa'],
    'bench50':  ['seq', 'par', 'hqa'],
    'bench100': ['seq', 'par', 'hqa'],
}
# Benchmark studies: circuit file and qubits per core (same hardware as test_compare.compare_w_sota)
BENCHMARKS = {
    'bench50':  dict(file='all_50.json',  qubits_per_core=10),
    'bench100': dict(file='all_100.json', qubits_per_core=10),
}


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
    _HAS_CUDA = torch.cuda.is_available() and (device is None or torch.device(device).type == 'cuda')

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


def _saveData(data, name):
    # Write to a temporary file first so a crash (or another node) never sees a half-written file
    tmp = f"{name}.tmp{os.getpid()}"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, name)


def _loadData(name):
    with open(name, 'r') as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------------------------
# Task definition. A task is a single circuit evaluated with both the sequential and parallel
# allocators, so the work can be spread over GPUs/nodes at the finest granularity.
# ---------------------------------------------------------------------------------------------

def _make_task(study, key, idx, nq, depth, n_cores, topology='all2all'):
    return dict(study=study, key=key, idx=idx, nq=nq, depth=depth, n_cores=n_cores, topology=topology)


def _build_tasks(studies, n_circs):
    tasks = []
    if 'depth' in studies:
        for d in [16*(2**i) for i in range(0,8)]:
            tasks += [_make_task('depth', d, i, nq=64, depth=d, n_cores=8) for i in range(n_circs)]
    if 'qubits' in studies:
        for q in [16*(2**i) for i in range(0,8)]:
            tasks += [_make_task('qubits', q, i, nq=q, depth=64, n_cores=8) for i in range(n_circs)]
    if 'cores' in studies:
        for c in [2*(2**i) for i in range(1,7)]:
            tasks += [_make_task('cores', c, i, nq=256, depth=64, n_cores=c) for i in range(n_circs)]
    if 'topology' in studies:
        for tn in TOPOLOGIES:
            tasks += [_make_task('topology', tn, i, nq=64, depth=32, n_cores=8, topology=tn) for i in range(n_circs)]
    # Benchmark circuits (all of them, n_circs does not apply) on every topology. The task idx is the circuit name
    for study, bench in BENCHMARKS.items():
        if study in studies:
            nq, circuits = _load_benchmark(study)
            for tn in TOPOLOGIES:
                tasks += [_make_task(study, tn, name, nq=nq, depth=len(slices), n_cores=nq//bench['qubits_per_core'], topology=tn)
                          for name, slices in circuits.items()]
    return tasks


@lru_cache
def _load_benchmark(study):
    ''' Returns (n_qubits, {circuit name: slices}) of the benchmark circuit file of the study. '''
    data = _loadData(os.path.join(DATA_DIR, BENCHMARKS[study]['file']))
    return data['n_qubits'], data['circuits']


def _build_circuit(task):
    if task['study'] in BENCHMARKS:
        nq, circuits = _load_benchmark(task['study'])
        return Circuit(slice_gates=circuits[task['idx']], n_qubits=nq)
    random.seed(_task_seed(task))
    return RandomCircuit(num_lq=task['nq'], num_slices=task['depth']).sample()


def _task_path(task):
    return os.path.join(PARTIAL_DIR, task['study'], f"{task['key']}_{task['idx']}.json")


def _task_seed(task):
    # Topologies share the same circuits so they can be compared pairwise
    if task['study'] == 'topology':
        return f"topology-{task['idx']}"
    return f"{task['study']}-{task['key']}-{task['idx']}"


def _task_cost_estimate(task):
    return task['nq']**2 * task['depth']


def _build_hardware(nq, n_cores, topology='all2all'):
    assert nq % n_cores == 0, f"{nq} qubits cannot be evenly split into {n_cores} cores"
    return Hardware(
        core_capacities=torch.tensor([nq//n_cores]*n_cores, dtype=torch.int),
        core_connectivity=TOPOLOGIES[topology](n_cores),
    )


def _run_allocator(allocator, circ, hardware, cfg, device):
    with torch.no_grad(), track_peak_memory(device) as mem:
        t0 = time()
        allocs, cost = allocator.optimize(circ, cfg=cfg, hardware=hardware)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t1 = time()
    # check_sanity raises on invalid cores / overflowed capacities, but returns the message for split gates
    err = check_sanity(allocs.cpu(), circ, hardware)
    if err:
        raise Exception(f"{type(allocator).__name__} returned an invalid allocation: {err}")
    res = dict(costs=cost/(circ.n_gates_norm + 1), times=t1-t0)
    for m in METRICS[2:]:
        res[m] = mem.get(m)
    return res


def _load_task(task):
    path = _task_path(task)
    return _loadData(path) if os.path.exists(path) else {}


def _missing_modes(task):
    done = _load_task(task)
    return [m for m in STUDY_MODES[task['study']] if m not in done]


def _run_task(task, allocators, device, modes):
    circ = _build_circuit(task)
    hardware = _build_hardware(task['nq'], task['n_cores'], task['topology'])
    da_cfg = DAConfig()
    # HQA runs on CPU, so no VRAM is tracked for it
    return {m: _run_allocator(allocators[m], circ, hardware, da_cfg, torch.device('cpu') if m == 'hqa' else device)
            for m in modes}


def _worker(rank, device_str, task_queue, n_threads):
    device = torch.device(device_str)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(n_threads)
    allocators = dict(
        seq=DirectAllocator.load(f'trained/{MODEL_NAME}', device_str, checkpoint=-1).set_mode(DirectAllocator.Mode.Sequential),
        par=DirectAllocator.load(f'trained/{MODEL_NAME}', device_str, checkpoint=-1).set_mode(DirectAllocator.Mode.Parallel),
        hqa=HQA(),
    )
    # Warm-up so CUDA context / kernel initialization is not charged to the first measured circuit
    _run_task(_make_task('warmup', 0, 0, nq=16, depth=8, n_cores=4), allocators, device, ['seq', 'par'])

    while True:
        task = task_queue.get()
        if task is None:
            break
        name = f"{task['study']}={task['key']} #{task['idx']}"
        print(f"[worker {rank} @ {device_str}] start {name}", flush=True)
        try:
            t0 = time()
            # Only the allocators missing from the task file are run, the rest of results are kept
            modes = _missing_modes(task)
            res = _load_task(task)
            res.update(_run_task(task, allocators, device, modes))
            os.makedirs(os.path.dirname(_task_path(task)), exist_ok=True)
            _saveData(res, _task_path(task))
            print(f"[worker {rank} @ {device_str}] done  {name} in {time()-t0:.1f}s", flush=True)
        except Exception:
            print(f"[worker {rank} @ {device_str}] FAILED {name}\n{traceback.format_exc()}", flush=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def _merge(studies, n_circs):
    ''' Assemble the per-circuit files of every complete study into the final json file.
    The lists of each key follow the task order: for benchmark studies, the order of the circuits in their file. '''
    tasks = _build_tasks(studies, n_circs)
    for study in studies:
        out_file = os.path.join(DATA_DIR, STUDY_FILES[study])
        study_tasks = [t for t in tasks if t['study'] == study]
        missing = [t for t in study_tasks if _missing_modes(t)]
        if missing:
            print(f"[merge] {study}: {len(missing)}/{len(study_tasks)} circuits missing, not writing {out_file}")
            continue
        data = {}
        for t in study_tasks:
            entry = data.setdefault(t['key'], {mode: defaultdict(list) for mode in STUDY_MODES[study]})
            res = _loadData(_task_path(t))
            for mode in STUDY_MODES[study]:
                for m in METRICS:
                    entry[mode][m].append(res[mode][m])
        _saveData(data, out_file)
        print(f"[merge] {study}: written {out_file}")


def _study_done(study):
    ''' The final file exists and contains the results of every allocator of the study. '''
    path = os.path.join(DATA_DIR, STUDY_FILES[study])
    return os.path.exists(path) and all(set(STUDY_MODES[study]) <= set(v) for v in _loadData(path).values())


def runStudies(studies, n_circs, n_gpus, num_nodes, node_rank):
    # Skip studies already in their final file, and circuits/allocators already computed (resumable)
    studies = [s for s in studies if not _study_done(s)]
    if not studies:
        print('All requested studies already computed')
        return
    tasks = _build_tasks(studies, n_circs)
    # Static split among nodes: interleave tasks sorted by estimated cost to balance the load.
    # Inside a node, GPUs pull tasks dynamically from a shared queue (heaviest first).
    tasks.sort(key=_task_cost_estimate, reverse=True)
    node_tasks = [t for j, t in enumerate(tasks) if j % num_nodes == node_rank]
    pending = [t for t in node_tasks if _missing_modes(t)]
    print(f"Node {node_rank}/{num_nodes}: {len(pending)} pending of {len(node_tasks)} tasks, studies={studies}")

    if n_gpus > 0:
        devices = [f'cuda:{i}' for i in range(n_gpus)]
    else:
        devices = ['cpu']
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpus = os.cpu_count()
    n_threads = max(1, n_cpus // len(devices))

    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    for t in pending:
        task_queue.put(t)
    for _ in devices:
        task_queue.put(None)
    procs = [ctx.Process(target=_worker, args=(r, d, task_queue, n_threads)) for r, d in enumerate(devices)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [r for r, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"Workers {failed} exited with errors")
    _merge(studies, n_circs)


def scalingAnalysis():
    depth_data = _loadData(os.path.join(DATA_DIR, STUDY_FILES['depth']))
    qubits_data = _loadData(os.path.join(DATA_DIR, STUDY_FILES['qubits']))
    cores_data = _loadData(os.path.join(DATA_DIR, STUDY_FILES['cores']))
    # TODO


def topologyAnalysis():
    topology_data = _loadData(os.path.join(DATA_DIR, STUDY_FILES['topology']))
    # TODO


def _parse_args():
    parser = argparse.ArgumentParser(description='Scalability and topology studies of the DirectAllocator')
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs of this node to use, one worker per GPU (0 = run on CPU). Default: all visible GPUs')
    parser.add_argument('--num-nodes', type=int, default=1,
                        help='Total number of nodes sharing the work (e.g. $SLURM_NNODES inside srun)')
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Index of this node in [0, num-nodes) (e.g. $SLURM_NODEID inside srun)')
    parser.add_argument('--studies', nargs='+', choices=list(STUDY_FILES), default=['depth', 'qubits', 'cores'],
                        help='Studies to compute')
    parser.add_argument('--n-circs', type=int, default=10, help='Circuits per configuration')
    parser.add_argument('--merge-only', action='store_true',
                        help='Do not compute, only assemble the per-circuit results into the final files')
    args = parser.parse_args()
    assert 0 <= args.gpus <= torch.cuda.device_count(), \
        f"Requested {args.gpus} GPUs but only {torch.cuda.device_count()} are visible"
    assert 0 <= args.node_rank < args.num_nodes, "node-rank must be in [0, num-nodes)"
    return args


if __name__ == '__main__':
    args = _parse_args()
    if args.merge_only:
        _merge(args.studies, args.n_circs)
    else:
        runStudies(args.studies, args.n_circs, args.gpus, args.num_nodes, args.node_rank)
