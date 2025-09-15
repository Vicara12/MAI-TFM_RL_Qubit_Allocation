#include "tree_search.hpp"
#include "inference_server.hpp"
#include <map>
#include <memory>
#include <optional>



struct TreeSearch::Node {
  // Key is the action (core allocation) and value are the resulting node (state) and action cost
  std::optional<std::map<int, std::tuple<std::shared_ptr<Node>, float>>> children = std::nullopt;
  bool terminal = false;
  float value_sum = 0;
  int visit_count = 0;
  int alloc_step = -1;
  int slice_idx = -1;
  at::Tensor prev_core_allocs = at::tensor({});
  at::Tensor current_core_allocs = at::tensor({});
  at::Tensor core_capacities = at::tensor({});
  at::Tensor policy = at::tensor({});


  auto expanded() const -> bool {return children.has_value();}


  auto value() const -> float {return (terminal? 0 : value_sum/visit_count);}


  auto get_child(int action) -> std::shared_ptr<Node> {
    if (children) {
      return std::get<0>(children->at(action));
    } else {
      throw std::runtime_error("Tried to get action child of a terminal node.");
    }
  }


  auto action_cost(int action) const -> float {
    if (children) {
      return std::get<1>(children->at(action));
    } else {
      throw std::runtime_error("Tried to get action cost of a terminal node.");
    }
  }
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
) -> std::tuple<at::Tensor, float> {
  slice_embs_ = slice_embs;
  circuit_embs_ = circuit_embs;
  alloc_steps_ = alloc_steps;
  cfg_ = std::forward(cfg);
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
  int num_sims = cfg_.target_tree_size - root_->visit_count;
  for (int i = 0; i < num_sims; i++) {
    std::vector<std::shared_ptr<Node>> search_path;
    search_path.push_back(root_);
    auto node = root_;

    while (node->expanded() and not node->terminal) {
      node = this->select_child(node);
      search_path.push_back(node);
    }

    if (not node->terminal)
      this->expand_node(node);
    this->backprop(search_path);
  }

  auto [action, logits] = TreeSearch::select_action(root_);
  // Remove node from children and assign to root variable all with move
  root_ = std::move(root_->get_child(action));
  return {action, logits, num_sims};
}


auto TreeSearch::select_action(std::shared_ptr<const Node> node) -> std::tuple<int, at::Tensor> {
    torch::Tensor visit_counts = torch::zeros({this->n_cores_}, torch::kFloat32);
    if (node->expanded()) {
      for (const auto& [action, v] : *(node->children)) {
        auto child_node = std::get<0>(v);
        visit_counts[action] = static_cast<float>(child_node->visit_count);
      }
    }

    float total_visits = visit_counts.sum().item<float>();
    if (total_visits > 0.0f) {
        visit_counts = visit_counts / total_visits;
    }

    int action = 0;
    if (cfg_.action_sel_temp == 0.0f) {
        action = visit_counts.argmax().item<int>();
    } else {
        torch::Tensor probs = torch::softmax(visit_counts / cfg_.action_sel_temp, /*dim=*/-1);
        action = torch::multinomial(probs, /*num_samples=*/1).item<int>();
    }

    return {action, visit_counts};
}


auto TreeSearch::new_policy_and_val(
  std::shared_ptr<const Node> node
) -> std::tuple<std::optional<at::Tensor>, float> {
  if (node->terminal)
    return {std::nullopt, 0.0f};

  auto qubits = torch::tensor(
    {alloc_steps_[node->alloc_step][1], alloc_steps_[node->alloc_step][2]}, torch::kI32);
  
  auto model_out = InferenceServer::pack_and_infer(
    "pred_model",
    qubits,
    node->prev_core_allocs,
    node->current_core_allocs,
    node->core_capacities,
    this->circuit_embs_[node->slice_idx],
    this->slice_ajdm_[node->slice_idx]
  );

  // Get unnormalized value
  float norm_val = model_out[1].item();
  float remaining_gates = alloc_steps_[node->alloc_step][3].item();
  float val = norm_val*(remaining_gates + 1);

  // Get policy (core allocation probabilities) with added noise
  at::Tensor pol = model_out[0];
  auto dir_noise = torch::distributions::Dirichlet(
    cfg_.dirichlet_alpha * torch::ones_like(pol)
  ).sample();
  pol = (1 - cfg_.noise)*pol + cfg_.noise*dir_noise;

  // Set probability of allocation of cores without enough space to zero and normalize
  int n_qubits = (qubits[1] == -1 ? 1 : 2);
  auto valid_moves_mask = node->core_capacities >= n_qubits;
  pol.index_put_(~valid_moves_mask, 0);
  float prob_sum = pol.sum();
  // If no probability mass (sum ~ 0) assign uniform probability to all valid moves
  if (prob_sum < 1e-5) {
    pol = at::zeros_like(pol);
    float n_valid_cores = valid_moves_mask.sum().item<float>();
    pol.index_put_(valid_moves_mask, 1/n_valid_cores);
  } else {
    pol /= prob_sum;
  }

  return {pol, val};
}


auto TreeSearch::build_root() -> Node {
  auto root = std::make_shared<Node>();
  root->current_core_allocs = -1*at::ones({core_caps_}, torch::kI32);
  root->prev_core_allocs    = -1*at::ones({core_caps_}, torch::kI32);
  root->core_capacities = core_caps_;
  root->alloc_step = 0;
  root->slice_idx  = 0;
  std::tie(root->policy, root->value_sum) = new_policy_and_val(root);
}


auto TreeSearch::expand_node(std::shared_ptr<Node> node) -> void {
  // TODO
}


auto TreeSearch::action_cost(std::shared_ptr<const Node> node, int action) -> float {
  // TODO
}


auto TreeSearch::backprop(std::vector<std::shared_ptr<Node>>& search_path) -> void {
  // TODO
}


auto TreeSearch::ucb(std::shared_ptr<const Node> node, int action) -> float {
  // TODO
}


auto TreeSearch::select_child(std::shared_ptr<const Node> current_node) -> std::shared_ptr<Node> {
  // TODO
}


auto TreeSearch::exploration_ratio(int n_exp_nodes) -> float {
  // TODO
}