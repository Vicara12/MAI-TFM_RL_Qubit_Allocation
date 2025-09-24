#pragma once
#include <torch/extension.h>
#include <vector>
#include <memory>



class TreeSearch
{
public:

  struct OptConfig {
    int target_tree_size = 1024;
    float noise = 0.25;
    float dirichlet_alpha = 0.3;
    float discount_factor = 1.0;
    float action_sel_temp = 0.0;
    float ucb_c1 = 1.25;
    float ucb_c2 = 19652;
  };

  struct TrainData {
    at::Tensor qubits;
    at::Tensor prev_allocs;
    at::Tensor curr_allocs;
    at::Tensor core_caps;
    at::Tensor slice_idx;
    at::Tensor logits;
    at::Tensor value;

    TrainData(int n_steps, int n_qubits, int n_cores);
  };

  TreeSearch(
    int n_qubits,
    const at::Tensor& core_capacities,
    const at::Tensor& core_conns
  );

  /**
   * @brief Optimize an entire circuit.
   *
   * @param slice_embs Tensor of shape [n_slices, n_qubits, n_qubits] with the slice adjacency matrices.
   * @param circuit_embs Tensor of shape [n_slices, n_qubits, n_qubits] with the per-slice circuit embeddings.
   * @param alloc_steps Tensor of shape [n_steps, 4] with, for each allocation step, the slice it
   * corresponds to, the two qubits involved (the second one must be set to -1 for single qubit
   * allocations) and the number of remaining gates to be allocated.
   * @param cfg OptConfig struct with the optimization parameters.
   * @param ret_train_data If true, a TrainData struct is returned with the data necesary to re-train the model.
   * @return Tuple with the allocation tensor of shape [n_qubits, n_slices] and the total number of expanded nodes, exploration ratio and optionally data for training.
   */
  auto optimize(
    const at::Tensor& slice_adjm,
    const at::Tensor& circuit_embs,
    const at::Tensor& alloc_steps,
    const OptConfig &cfg,
    bool ret_train_data
  ) -> std::tuple<at::Tensor, int, float, std::optional<TrainData>>;

  
private:

  struct Node;
  int n_qubits_;
  int n_cores_;
  int n_steps_;
  at::Tensor core_caps_;
  at::Tensor core_conns_;
  // These are initialized at training
  at::Tensor slice_adjm_;
  at::Tensor circuit_embs_;
  at::Tensor alloc_steps_;
  OptConfig cfg_;
  std::shared_ptr<Node> root_;


  auto store_train_data(
    TrainData& tdata, int alloc_step, int slice_idx, int q0, int q1
  ) -> void;

  auto initialize_search(
    const at::Tensor& slice_adjm,
    const at::Tensor& circuit_embs,
    const at::Tensor& alloc_steps,
    const OptConfig &cfg
  ) -> at::Tensor;

  auto iterate() -> std::tuple<int, float, at::Tensor, int>;

  auto select_action(
    std::shared_ptr<const Node> node
  ) -> std::tuple<int, at::Tensor>;

  auto new_policy_and_val(std::shared_ptr<const Node> node) -> std::tuple<std::optional<at::Tensor>, float>;

  auto build_root() -> std::shared_ptr<Node>;

  auto expand_node(std::shared_ptr<Node> node) -> void;

  auto action_cost(std::shared_ptr<const Node> node, int action) -> float;

  auto backprop(std::vector<std::tuple<std::shared_ptr<Node>, int>>& search_path) -> void;

  auto ucb(std::shared_ptr<const Node> node, int action) -> float;

  auto select_child(std::shared_ptr<const Node> current_node) -> std::tuple<std::shared_ptr<Node>, int>;

  auto exploration_ratio(int n_exp_nodes) -> float;

};
