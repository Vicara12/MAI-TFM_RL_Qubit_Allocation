#include "tree_search.h"
#include <map>
#include <memory>
#include <optional>



struct TreeSearch::Node {
  // Key is the action (core allocation) and value are the resulting node (state) and action cost
  std::optional<std::map<int, std::tuple<std::unique_ptr<Node>, float>>> children = std::nullopt;
  bool terminal = false;
  float value_sum = 0;
  int visit_count = 0;

  auto Node::expanded() -> bool {return children.has_value();}

  auto Node::value() -> float {return (terminal? 0 : value_sum/visit_count);}
};


TreeSearch::TreeSearch(
  int n_qubits,
  const at::Tensor& core_capacities,
  const at::Tensor& core_conns
)
  : n_qubits_(n_qubits)
  , core_caps_(core_capacities)
  , core_conns_(core_conns)
  , n_cores_(core_conns.size(0))
{}


auto TreeSearch::optimize_for_train(
  const at::Tensor& slice_embs,
  const at::Tensor& circuit_embs,
  const at::Tensor& alloc_steps,
  auto&& cfg
) -> std::tuple<at::Tensor, float, at::Tensor> {
  auto n_slices = slice_embs.size(0);
  auto n_steps = alloc_steps.size(0);
  at::Tensor allocs = at::empty({n_qubits_, n_slices}, at::dtype(at::kInt));
  int n_sims = 0;
  at::Tensor sel_probs = at::empty({n_steps, n_cores_}, at::dtype(at::kFloat));

  for (int64_t step = 0; step < n_steps; step++) {
    // TODO
  }

  return {allocs, n_sims, sel_probs};
}


auto TreeSearch::optimize(
  const at::Tensor& slice_embs,
  const at::Tensor& circuit_embs,
  const at::Tensor& alloc_steps,
  auto&& cfg
) -> std::tuple<at::Tensor, float> {
  // TODO
}


TreeSearch::~TreeSearch() {
  // TODO: clean ts
}


auto TreeSearch::iterate() -> std::tuple<int, at::Tensor, int> {
  // TODO
}


auto TreeSearch::select_action(const Node& node, float temp) -> int {
  // TODO
}


auto TreeSearch::new_policy_and_val(const Node& node) -> std::tuple<at::Tensor, float> {
  // TODO
}


auto TreeSearch::build_root() -> Node {
  // TODO
}


auto TreeSearch::expand_node(Node& node) -> void {
  // TODO
}


auto TreeSearch::action_cost(const Node& node, int action) -> float {
  // TODO
}


auto TreeSearch::backprop(const std::vector<const Node&>& search_path) -> void {
  // TODO
}


auto TreeSearch::ucb(const Node& node, int action) -> float {
  // TODO
}


auto TreeSearch::select_child(const Node& current_node) -> std::tuple<int, Node&> {
  // TODO
}


auto TreeSearch::exploration_ratio(int n_exp_nodes) -> float {
  // TODO
}