#pragma once
#include <torch/extension.h>
#include <vector>



class TreeSearch
{
  typedef struct Node;
  int n_qubits_;
  int n_cores_;
  at::Tensor core_caps_;
  at::Tensor core_conns_;

  auto iterate() -> std::tuple<int, at::Tensor, int>;

  auto select_action(const Node& node, float temp) -> int;

  auto new_policy_and_val(const Node& node) -> std::tuple<at::Tensor, float>;

  auto build_root() -> Node;

  auto expand_node(Node& node) -> void;

  auto action_cost(const Node& node, int action) -> float;

  auto backprop(const std::vector<const Node&>& search_path) -> void;

  auto ucb(const Node& node, int action) -> float;

  auto select_child(const Node& current_node) -> std::tuple<int, Node&>;

  auto exploration_ratio(int n_exp_nodes) -> float;

public:

  typedef struct {
    int target_tree_size = 1024;
    float noise = 0.25;
    float dirichlet_alpha = 0.3;
    float discount_factor = 1.0;
    float action_sel_temp = 0.0;
    float ucb_c1 = 1.25;
    float ucb_c2 = 19652;
  } OptConfig;

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
   * @return Tuple with the allocation tensor of shape [n_qubits, n_slices] and the exploration ratio.
   */
  auto optimize(
    const at::Tensor& slice_embs,
    const at::Tensor& circuit_embs,
    const at::Tensor& alloc_steps,
    auto&& cfg // OptConfig
  ) -> std::tuple<at::Tensor, float>;

  /**
   * 
   */
  auto optimize_for_train(
    const at::Tensor& slice_embs,
    const at::Tensor& circuit_embs,
    const at::Tensor& alloc_steps,
    auto&& cfg // OptConfig
  ) -> std::tuple<at::Tensor, float>;

  ~TreeSearch();
};
