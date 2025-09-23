from typing import Optional, Union

import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from generator import CircuitGenerator
from render import render
#from .render import render

log = get_pylogger(__name__)


class QubitAllocEnv(RL4COEnvBase):
    """Qubit allocation environment.
    Environment for the multi-core qubit mapping problem, where qubits must be assigned to cores across multiple time slices,
    respecting core capacities and qubit coupling constraints. Here it's framed as a multi-agent problem, where each qubit is an agent,
    and each node is a core. For each time slice, agents select cores in a single round but, if they choose conflicting actions, 
    the conflict handler is called and agents that lose the conflict must reselect from the remaining available cores in the next round
    until all agents have selected a core. Only then the environment state transitions to the next time slice.

    Observations:


    Constraints:


    Finish Condition:

    Reward:
        - The reward is the negative of the total cost, 

    Args:
        generator: An instance of CircuitGenerator used as the data generator for quantum circuits
        generator_params: Parameters configuring the generator
    """

    name = "qubit_allocation"

    def __init__(
        self,
        generator: Union[CircuitGenerator, None] = None,
        generator_params: dict = {},
        num_cores: int = 2,
        core_capacity: int = 4,
        distance_matrix: torch.Tensor = None,
        check_solution: bool = False,
        **kwargs,
    ):
        kwargs["check_solution"] = check_solution
        if kwargs["check_solution"]:
            log.warning(
                "Check solution is enabled, this will slow down the training/testing and should be used for debugging purposes only."
            )
        super().__init__(**kwargs)

        if distance_matrix is None:  # All to all architecture
            distance_matrix = torch.ones((num_cores, num_cores), dtype=torch.float32, requires_grad=False) \
                                - torch.eye(num_cores, dtype=torch.float32, requires_grad=False)
        self.distance_matrix = distance_matrix
        self.num_cores = num_cores
        self.core_capacity = core_capacity  # Assume all cores have the same capacity

        if generator is None:
            generator = CircuitGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)


    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        """
        Returns:
            A TensorDict containing the following keys:
        """
        device = td.device

        init_slices = getattr(td, "slices", None)
        #if batch_size is None:
            #batch_size = self.batch_size if init_slices is None else init_slices.shape[0]
        device = init_slices.device if init_slices is not None else self.device
        self.to(device)

        init_slices = td["slices"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize state variables
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_slice = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_core_capacity = torch.full(
            (*batch_size, self.num_cores), self.core_capacity, dtype=torch.int64, device=device
        )
        current_assignment = torch.full((*batch_size, self.num_qubits), self.num_cores, dtype=torch.int64, device=device)
        reward = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)

        # Init action mask
        action_mask = torch.ones(
            (*batch_size, self.num_qubits, self.num_cores), dtype=torch.bool, device=device
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "slices": init_slices,
                "i": i, 
                "current_slice": current_slice,
                "current_core_capacity": current_core_capacity,
                "reward": reward,
                "done": torch.zeros((*batch_size, 1), dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Keys:
            - action [batch_size, num_agents]: action taken by each agent
        """

        current_slice = td['current_slice']
        batch_size = td.batch_size[0] if isinstance(td.batch_size, (tuple, list)) else td.batch_size
        num_qubits = self.generator.num_qubits

        chosen_cores = td['action']  # shape: (batch_size, num_qubits)
        current_core_capacity = td["current_core_capacity"]  # shape: (batch_size, num_cores)
        current_assignment = td['current_assignment']  # shape: (batch_size, num_qubits)

        # Update core capacities
        assigned_counts = torch.zeros_like(current_core_capacity)
        assigned_counts.scatter_add_(1, chosen_cores, torch.ones_like(chosen_cores, dtype=current_core_capacity.dtype))
        current_core_capacity -= assigned_counts

        assert (current_core_capacity >= 0).all(), "Negative core capacity after allocation"
        #TODO: change assertion to episode truncation

        # Update assignments
        current_assignment[:] = chosen_cores

        # Prepare for next slice or reset if done
        last_assignment = current_assignment

        if current_slice[0] == self.num_slices - 1:
            # Last slice
            current_assignment = torch.full(
                (batch_size, num_qubits), self.num_cores, dtype=torch.int64, device=self.device
            )
            current_core_capacity = torch.full(
                (batch_size, self.num_cores), self.core_size, dtype=torch.float32, device=self.device
            )
            # Done is only True when all qubits have been assigned in the last slice
            done = (current_assignment != self.num_cores).all(dim=-1, keepdim=True)
        else:
            done = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
            # Not always. Only if all qubits have been assigned
            if (current_assignment != self.num_cores).all():
                current_slice = current_slice + 1

        reward = torch.zeros_like(done)

        td.update(
            {
                "current_slice": current_slice,
                "current_core_capacity": current_core_capacity,
                "current_assignment": current_assignment,
                "last_assignment": last_assignment,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """
        Action mask for each agent (qubit) to select a core.
        We mask when:
            1. The core is already at full capacity (capacity < 1)
            2. The core has capacity 2 but the qubit has a friend (capacity < 2)
            3. The qubit has already been allocated (assigned != num_cores)
        Args:
            td: TensorDict containing the current state
        """
        batch_size = td.batch_size
        num_qubits = td["current_capacity"].size(-1)
        num_cores = td["current_capacity"].size(-1)

        indices = td["slice"]._indices()  # [4, num_pairs]

        core_cap = td["current_core_capacity"]  
        action_mask = (core_cap.unsqueeze(1) >= 1).expand(batch_size, num_qubits, num_cores).clone()

        # Find (batch, qubit) pairs with a friend (friend != -1)
        has_friend_mask = indices[3] != -1
        if has_friend_mask.any():
            # Get all with a friend
            b_idx = indices[0, has_friend_mask]
            q_idx = indices[2, has_friend_mask]
            # For these, only allow cores with capacity >= 2
            action_mask[b_idx, q_idx] = core_cap[b_idx] >= 2

        # Mask out actions for qubits that have already been allocated
        current_assignment = td["current_assignment"]  
        allocated_mask = current_assignment != num_cores  # [batch, num_qubits]
        action_mask[allocated_mask] = False

        #TODO: Imagine q1 and q2 are pairs. In the previous round, q1 chose core 0 and q2 chose core 1.
        # The conflict handler was called and q1 won the conflict. Now, in this round, q2 must reselect a core.
        # All cores which aren't core 0 must be masked out for q2. 
        # If there is only capacity 1 in this core, then we should mask core 0 for all other qubits as well.

        return action_mask


    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:

        self.distance_matrix = self.distance_matrix.to(td.device)
        src = actions[:, 1:]    # [batch_size, num_steps-1]
        dst = actions[:, :-1]   # [batch_size, num_steps-1]
        cost = self.distance_matrix[src, dst].sum(dim=1)  # [batch_size]
         
        if self.normalize_reward:
            # Count how many times each batch index appears in the slices indices
            batch_indices = td['slices']._indices()[0]  # [N]
            num_gates = torch.bincount(batch_indices, minlength=cost.shape[0])  # [batch_size]
            cost = cost / num_gates.clamp(min=1)  # avoid division by zero

        return -cost  

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check the validity of the solution.

        Notes:
            - This function is implemented in a low efficiency way, only for debugging purposes.
        """
        num_agents = td["current_node"].size(-1)
        num_loc = td["locs"].size(-2) - num_agents
        batch_size = td.batch_size

        # Flatten the actions of all agents
        actions_flatten = actions.flatten(start_dim=-2)

        # Sort the actions from small to large
        actions_flatten_sort = actions_flatten.sort(dim=-1)[0]

        # Check if visited all nodes
        for batch_idx in range(*batch_size):
            actions_sort_unique = torch.unique(actions_flatten_sort[batch_idx])
            actions_sort_unique = actions_sort_unique[actions_sort_unique >= num_agents]
            assert (
                torch.arange(num_agents, num_agents + num_loc, device=td.device)
                == actions_sort_unique
            ).all(), f"Invalid tour at batch {batch_idx} with tour {actions_sort_unique}"

        # TODO: double check the validity of the demand

    def _make_spec(self, generator: CircuitGenerator):
        self.observation_spec = Composite(
            locs=Unbounded(
                shape=(self.generator.num_slices, self.generator.num_qubits, self.generator.num_qubits),
                dtype=torch.bool,
            ),
            current_slice=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(self.num_cores + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.num_cores - 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def render(self, td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax, distance_matrix=self.distance_matrix)