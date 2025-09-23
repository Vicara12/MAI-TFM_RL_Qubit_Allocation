import random
import time
from typing import Callable, Optional, Union

import torch

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform
import numpy as np

from numba import jit

log = get_pylogger(__name__)


class CircuitGenerator(Generator):
    def __init__(
        self,
        num_slices: int = 16,
        num_qubits: int = 8,
        seed: Optional[Union[int, None]] = None,
        **kwargs,
    ):
        self.num_slices = num_slices
        self.num_qubits = num_qubits
        self.seed = seed

    @staticmethod   
    @jit(nopython=True, cache=True, parallel=False)
    def _fast_generator(rng, num_qubits, num_slices, batch_size):
        edges = np.argwhere(~np.eye(num_qubits, dtype=np.bool_)).astype(np.int16)        
        indices = []
        
        for i in range(batch_size):
            slices = np.full((num_slices, num_qubits), False)
            t = 0
            while t < num_slices - 1:
                q1, q2 = edges[rng.integers(low=0, high=edges.shape[0])]
                for t_ in range(t, -1, -1):
                    if slices[t_][q1] or slices[t_][q2]:
                        t_ += 1
                        slices[t_][q1] = slices[t_][q2] = True
                        indices.extend([(i,t_,q1,q2), (i,t_,q2,q1)])
                        break
                if t_ == 0:
                    slices[0][q1] = slices[0][q2] = True
                    indices.extend([(i,0,q1,q2), (i,0,q2,q1)])
                else:
                    t = max(t, t_)
            
        return np.array(indices).T
    
        
    def _generate(self, batch_size) -> TensorDict:

        #TODO: Remember to comment to randomize training data
        if self.seed is None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng(self.seed)
            self.seed += 1
        time_start = time.perf_counter()

        indices = self._fast_generator(rng, self.num_qubits, self.num_slices, *batch_size)
        slices = torch.sparse_coo_tensor(indices, np.ones(indices.shape[1], dtype=bool), 
                                       (*batch_size, self.num_slices, self.num_qubits, self.num_qubits))

        print('Generated dataset with {} in {:.3f} s'.format(batch_size, time.perf_counter() - time_start))

        return TensorDict(
            {
                "slices": slices
            },
            batch_size=batch_size,
        )