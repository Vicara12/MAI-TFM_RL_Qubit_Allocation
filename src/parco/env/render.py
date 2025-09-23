import numpy as np
from torch import Tensor
from typing import Optional
from tensordict import TensorDict
from rl4co.utils.pylogger import get_pylogger
import plotly.graph_objects as go
import plotly.express as px

log = get_pylogger(__name__)

def render(td: TensorDict, actions: Optional[Tensor]=None, ax=None, distance_matrix=None):
    """Render the mapping solution as a sankey diagram. If actions is None, render the current state."""
    
    if actions is None:
        actions = td.get("action", None)

    num_slices = td.get("num_slices", None)
    num_cores = td.get("current_core_capacity", None).shape[-1]
    num_qubits = td.get("current_assignments", None).shape[-1]

    # render only the first batch element
    data = actions[0].view(num_slices, num_qubits).cpu().numpy()
    
    #labels = [f'{i // self.num_cores}-{i % self.num_cores:02d}' for i in range(data.shape[0])]
    labels = [f'{s:02d}-{c:02d}' for s in range(num_slices) for c in range(num_cores)]
    
    cx = [(s+0.01) / (num_slices-0.99) for s in range(num_slices) for c in range(num_cores) ]
    cy = [(c+0.01) / (num_cores-0.99) for s in range(num_slices) for c in range(num_cores) ]

    core_colors = px.colors.sample_colorscale('turbo', [c / (num_cores-1) for c in range(num_cores)])
    max_distance = distance_matrix.max().long().item()
    edge_colors = px.colors.sample_colorscale('matter', [d / (max_distance) for d in range(max_distance+1)])
    colors = core_colors * num_slices

    sources, targets, values, link_colors, link_labels = [], [], [], [], []
    for s in range(1,num_slices):
        for cs in range(num_cores):
            for ct in range(num_cores):
                sources.append((s-1)*num_cores + cs)
                targets.append(s*num_cores + ct)
                values.append(len(set(np.where(data[s] == ct)[0]).intersection(set(np.where(data[s-1] == cs)[0]))))
                distance = distance_matrix[cs,ct].long().item()
                link_labels.append(str(distance))
                link_colors.append(f'rgba{edge_colors[distance][3:-1]}, {0.4})') 
                
    
    fig = go.Figure(data=[go.Sankey(
        arrangement = "fixed",
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = colors,
            # TODO fix pos
            x = cx,
            y = cy
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values,
            color = link_colors,
            label = link_labels,
            #arrowlen=10,
    ))])

    fig.update_layout(title_text="Qubit flow", font_size=10)
    fig.show()
    
    return fig