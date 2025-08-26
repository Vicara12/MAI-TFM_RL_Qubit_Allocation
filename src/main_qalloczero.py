import torch
from utils.timer import Timer
from utils.allocutils import solutionCost
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.mcts import MCTS
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.models.enccircuit import CircuitEncoder
from qalloczero.models.snapshotenc import SnapEncModel
from qalloczero.models.predmodel import PredictionModel
from qalloczero.models.inferenceserver import InferenceServer
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation



def main():
  q_emb_size = 16
  encoder_shape = (16, 8, 8)
  core_caps = torch.tensor([4,4,4], dtype=int)
  core_con = torch.ones(size=(len(core_caps),len(core_caps)), dtype=int) - torch.eye(n=len(core_caps), dtype=int)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_con)
  q_embs = torch.nn.Parameter(torch.randn(hardware.n_physical_qubits, q_emb_size), requires_grad=True)
  dummy_q_emb = torch.nn.Parameter(torch.randn(q_emb_size), requires_grad=True)

  # circuit_encoder = GNNEncoder(
  #   hardware=hardware,
  #   nn_dims=encoder_shape,
  #   qubit_embs=q_embs
  # )
  circuit_encoder = None
  snap_enc = SnapEncModel(
    nn_dims=(16,8),
    hardware=hardware,
    qubit_embs=q_embs,
    dummy_qubit_emb=dummy_q_emb
  )
  pred_mod = PredictionModel(
     config=PredictionModel.Config(hw=hardware, circuit_emb_shape=8, mha_num_heads=4),
     qubit_embs=q_embs,
  )

  InferenceServer.addModel('circ_enc', circuit_encoder)
  InferenceServer.addModel("snap_enc", snap_enc)
  InferenceServer.addModel("pred_model", pred_mod)

  sampler = RandomCircuit(num_lq=sum(core_caps).item(), num_slices=10)
  mcts_config = MCTS.Config(target_tree_size=1024)

  azero_config = AlphaZero.Config(
    hardware=hardware,
    encoder_shape=encoder_shape,
    mcts_config=mcts_config
  )
  azero_train_cfg = AlphaZero.TrainConfig(
    train_iters=100,
    batch_size=3,
    sampler=sampler,
    lr=0.01,
    v_weight=1,
    logit_weight=1
  )
  azero = AlphaZero(config=azero_config, qubit_embs=q_embs)
  with Timer.get('t0'):
    azero.train(azero_train_cfg)

  # cost = solutionCost(allocations,hardware.core_connectivity)
  # print(f" -> cost={cost} time={Timer.get('t0').total_time} e_ratio={exploration_ratio}")
  # drawQubitAllocation(allocations, core_caps, circuit.slice_gates)



def testing():
  sampler = RandomCircuit(num_lq=4, num_slices=3)
  cs = [sampler.sample().adj_matrices.unsqueeze(0) for _ in range(2)]
  encoder = CircuitEncoder(n_qubits=4, n_heads=4, n_layers=4)
  encoder.eval()
  embs = [encoder(m) for m in cs]
  embs_b = encoder(torch.vstack(cs))
  print(torch.equal(torch.vstack(embs), embs_b))


if __name__ == "__main__":
  # main()
  testing()