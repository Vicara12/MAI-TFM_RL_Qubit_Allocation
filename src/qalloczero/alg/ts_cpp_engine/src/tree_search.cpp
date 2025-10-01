#include "tree_search.hpp"
#include <map>
#include <memory>
#include <optional>
#include <cassert>
#include <cmath>

using Slice = at::indexing::Slice;



struct TreeSearch::Node {
  // Key is the action (core allocation) and value are the resulting node (state) and action cost
  std::optional<std::map<int, std::tuple<std::shared_ptr<Node>, float>>> children = std::nullopt;
  bool terminal = false;
  float value_sum = 0;
  int visit_count = 0;
  int alloc_step = -1;
  int slice_idx = -1;
  std::optional<at::Tensor> prev_allocs;
  std::optional<at::Tensor> current_allocs;
  std::optional<at::Tensor> core_caps;
  std::optional<at::Tensor> policy;

  auto print(int rec = 0, bool simple = false) const -> void {
    std::cout << "NODE: " << this << std::endl
              << " - children:" << std::endl;
    if (children.has_value())
      for (const auto& [action, node] : *children)
        std::cout << "   + action " << action << " node " << std::get<0>(node) << " cost " << std::get<1>(node) << std::endl;
    else
      std::cout << "   None" << std::endl;
    std::cout << " - terminal: " << (terminal ? "true" : "false") << std::endl
              << " - value_sum: " << value_sum << std::endl
              << " - visit count: " << visit_count << std::endl
              << " - alloc_step: " << alloc_step << std::endl
              << " - slice_idx: " << slice_idx << std::endl;

    if (not simple) {
      if (prev_allocs.has_value())
        std::cout << " - prev_allocs: " << *prev_allocs << std::endl;
      else
        std::cout << " - prev_allocs: none" << std::endl;
      
      if (current_allocs.has_value())
        std::cout << " - current_allocs: " << *current_allocs << std::endl;
      else
        std::cout << " - current_allocs: none" << std::endl;
      
      if (core_caps.has_value())
        std::cout << " - core_caps: " << *core_caps << std::endl;
      else
        std::cout << " - core_caps: none" << std::endl;
      
      if (policy.has_value())
        std::cout << " - policy: " << *policy << std::endl;
      else
        std::cout << " - policy: none" << std::endl;
    }
    
    if (rec != 0 and children.has_value()) {
      for (const auto& [_, node] : *children)
        std::get<0>(node)->print(rec-1, simple);
    }
  }


  auto expanded() const -> bool {return children.has_value();}

  auto value() const -> float {return (terminal ? 0 : value_sum/(visit_count+1));}

  auto get_child(int action) const -> std::shared_ptr<Node> {
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


struct TreeSearch::OptCtx {
  int n_steps_;
  int n_slices_;
  at::Tensor core_caps_;
  at::Tensor core_conns_;
  at::Tensor slice_adjm_;
  at::Tensor circuit_embs_;
  at::Tensor alloc_steps_;
  OptConfig cfg_;
  std::shared_ptr<Node> root_;
  std::optional<size_t> inference_ctx;
};


TreeSearch::TrainData::TrainData(int n_steps, int n_qubits, int n_cores, at::Device device)
  : qubits(at::empty({n_steps, 2}, at::TensorOptions().dtype(at::kInt).device(device)))
  , prev_allocs(at::empty({n_steps, n_qubits}, at::TensorOptions().dtype(at::kLong).device(device)))
  , curr_allocs(at::empty({n_steps, n_qubits}, at::TensorOptions().dtype(at::kLong).device(device)))
  , core_caps(at::empty({n_steps, n_cores}, at::TensorOptions().dtype(at::kLong).device(device)))
  , slice_idx(at::empty({n_steps, 1}, at::TensorOptions().dtype(at::kLong).device(device)))
  , logits(at::empty({n_steps, n_cores}, at::TensorOptions().dtype(at::kFloat).device(device)))
  , value(at::empty({n_steps, 1}, at::TensorOptions().dtype(at::kFloat).device(device)))
{}


TreeSearch::TreeSearch(
  int n_qubits,
  int n_cores,
  at::Device device
)
  : n_qubits_(n_qubits)
  , n_cores_(n_cores)
  , device_(device)
  , parallel_opt_counter(0)
{}

auto TreeSearch::get_is() -> InferenceServer& {
  return is_;
}


auto TreeSearch::optimize(
  const at::Tensor& core_conns,
  const at::Tensor& core_caps,
  const at::Tensor& slice_adjm,
  const at::Tensor& circuit_embs,
  const at::Tensor& alloc_steps,
  const OptConfig &cfg,
  bool ret_train_data,
  bool verbose
) const -> std::tuple<at::Tensor, int, float, std::optional<TrainData>> {
  auto ctx = initialize_search(
    core_conns, core_caps, slice_adjm, circuit_embs, alloc_steps, cfg);
  at::Tensor allocs = at::empty({ctx.n_slices_, n_qubits_}, at::kInt);
  std::optional<TrainData> tdata;
  if (ret_train_data)
    tdata = TrainData(ctx.n_steps_, n_qubits_, n_cores_, device_);
  int n_expanded_nodes = 0;

  for (size_t step = 0; step < ctx.n_steps_; step++) {
    int slice_idx = ctx.alloc_steps_[ctx.root_->alloc_step][0].item<int>();
    int qubit0 = ctx.alloc_steps_[ctx.root_->alloc_step][1].item<int>();
    int qubit1 = ctx.alloc_steps_[ctx.root_->alloc_step][2].item<int>();

    if (verbose)
      std::cout << " - Optimization step " << (step+1) << "/" << ctx.n_steps_ << std::endl;

    if (ret_train_data)
      store_train_data(ctx, *tdata, step, slice_idx, qubit0, qubit1);

    auto [action, action_cost, logits, n_sims] = iterate(ctx);

    n_expanded_nodes += n_sims;
    allocs[slice_idx][qubit0] = action;
    if (qubit1 != -1)
      allocs[slice_idx][qubit1] = action;
    
    if (ret_train_data) {
      tdata->logits.index_put_({int(step)}, logits);
      tdata->value[step][0] = action_cost;
    }
  }

  if (ret_train_data) {
    // Compute total remaining cost for each step and normalize
    for (int step = ctx.n_steps_-2; step >= 0; step--) {
      tdata->value[step][0] += tdata->value[step+1][0];
      float rem_gates = ctx.alloc_steps_[step+1][3].item<float>();
      tdata->value[step+1][0] /= (rem_gates+1);
    }
    float rem_gates = ctx.alloc_steps_[0][3].item<float>();
    tdata->value[0][0] /= (rem_gates+1);
  }

  float expl_r = exploration_ratio(ctx, n_expanded_nodes);

  if (ctx.inference_ctx.has_value()) {
    is_.rm_context(*ctx.inference_ctx);
  }

  return {allocs, n_expanded_nodes, expl_r, tdata};
}


auto TreeSearch::store_train_data(
  const OptCtx &ctx,
  TrainData& tdata,
  int alloc_step,
  int slice_idx,
  int q0,
  int q1
) const -> void {
  tdata.qubits[alloc_step][0] = q0;
  tdata.qubits[alloc_step][1] = q1;
  tdata.slice_idx[alloc_step][0] = slice_idx;
  tdata.prev_allocs.index_put_({alloc_step}, *ctx.root_->prev_allocs);
  tdata.curr_allocs.index_put_({alloc_step}, *ctx.root_->current_allocs);
  tdata.core_caps.index_put_({alloc_step}, *ctx.root_->core_caps);
}


auto TreeSearch::initialize_search(
  const at::Tensor& core_conns,
  const at::Tensor& core_caps,
  const at::Tensor& slice_adjm,
  const at::Tensor& circuit_embs,
  const at::Tensor& alloc_steps,
  const OptConfig &cfg
) const -> OptCtx {
  OptCtx ctx;
  // Executing a torch model is not thread safe, so what we do is to have a parallel optimization
  // counter that tells how many optimizations are taking place in parallel. If there's none then
  // there is no need to create a new inference context. If there is an optimization going on we
  // create a new inference context so that it does not mess with the previous one.
  int n_opts = parallel_opt_counter.fetch_add(1);
  if (n_opts > 0) {
    // We use as key the address of the ctx variable to prevent concurrent duplicates
    ctx.inference_ctx = size_t(&ctx);
    is_.new_context(*ctx.inference_ctx);
  }
  ctx.core_conns_ = core_conns.to(device_);
  ctx.core_caps_ = core_caps.to(device_);
  ctx.slice_adjm_ = slice_adjm.to(device_);
  ctx.circuit_embs_ = circuit_embs.to(device_);
  ctx.alloc_steps_ = alloc_steps;
  ctx.cfg_ = cfg;
  ctx.n_slices_ = slice_adjm.size(0);
  ctx.n_steps_ = alloc_steps.size(0);
  ctx.root_ = build_root(ctx);
  return ctx;
}


auto TreeSearch::iterate(OptCtx &ctx) const -> std::tuple<int, float, at::Tensor, int> {
  int num_sims = ctx.cfg_.target_tree_size - ctx.root_->visit_count;
  for (int i = 0; i < num_sims; i++) {
    // Node and action that lead to that node pairs
    std::vector<std::tuple<std::shared_ptr<Node>, int>> search_path;
    search_path.push_back({ctx.root_, -1});
    int action = -1;
    auto node = ctx.root_;

    while (node->expanded() and not node->terminal) {
      std::tie(node, action) = this->select_child(ctx, node);
      search_path.push_back({node, action});
    }

    if (not node->terminal)
      this->expand_node(ctx, node);
    this->backprop(search_path, ctx.cfg_.discount_factor);
  }

  auto [action, logits] = TreeSearch::select_action(ctx.root_, ctx.cfg_.action_sel_temp);
  float action_cost = ctx.root_->action_cost(action);
  // Remove node from children and assign to root variable all with move
  ctx.root_ = std::move(ctx.root_->get_child(action));
  return {action, action_cost, logits, num_sims};
}


auto TreeSearch::select_action(
  std::shared_ptr<const Node> node,
  float temp
) const -> std::tuple<int, at::Tensor> {
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
    if (temp == 0.0f) {
        action = visit_counts.argmax().item<int>();
    } else {
        torch::Tensor probs = torch::softmax(visit_counts / temp, /*dim=*/-1);
        action = torch::multinomial(probs, /*num_samples=*/1).item<int>();
    }

    return {action, visit_counts};
}


auto TreeSearch::new_policy_and_val(
  const OptCtx &ctx,
  int q0,
  int q1,
  int remaining_gates,
  int slice_idx,
  const at::Tensor &prev_allocs, // [Q]
  const at::Tensor &curr_allocs, // [B,Q]
  const at::Tensor &core_caps    // [B,C]
) const -> std::tuple<at::Tensor, at::Tensor> {
  auto qubits = at::tensor({q0, q1}, torch::kInt32).to(device_);
  int batch = curr_allocs.size(0);

  auto model_out = is_.infer(
    ctx.inference_ctx,
    qubits.expand({batch,2}),
    prev_allocs.expand({batch,n_qubits_}),
    curr_allocs,
    core_caps,
    ctx.core_conns_,
    ctx.circuit_embs_.index({slice_idx, Slice(), Slice()}).expand({batch, n_qubits_, n_qubits_}),
    ctx.slice_adjm_.index({slice_idx, Slice(), Slice()}).expand({batch, n_qubits_, n_qubits_})
  );

  // Get unnormalized value
  at::Tensor val = model_out[1]*(remaining_gates + 1);

  // Get policy (core allocation probabilities) with added noise
  at::Tensor pol = model_out[0];
  auto dir_noise = at::_sample_dirichlet(
    ctx.cfg_.dirichlet_alpha * torch::ones_like(pol));
  pol = (1 - ctx.cfg_.noise)*pol + ctx.cfg_.noise*dir_noise;

  // Set probability of allocation of cores without enough space to zero and normalize
  int n_qubits = (q1 == -1 ? 1 : 2);
  auto valid_moves_mask = core_caps >= n_qubits;
  pol.index_put_({~valid_moves_mask}, at::tensor(0.f));
  at::Tensor prob_sum = pol.sum(-1).unsqueeze(1);
  pol /= prob_sum;

  return {pol, val};
}


auto TreeSearch::build_root(const OptCtx &ctx) const -> std::shared_ptr<Node> {
  auto root = std::make_shared<Node>();
  root->current_allocs = n_cores_*at::ones({n_qubits_}, torch::kLong).to(device_);
  root->prev_allocs    = n_cores_*at::ones({n_qubits_}, torch::kLong).to(device_);
  root->core_caps = ctx.core_caps_;
  root->alloc_step = 0;
  root->slice_idx  = 0;
  auto [pol, val] = new_policy_and_val(
    ctx,
    ctx.alloc_steps_[0][1].item<int>(),
    ctx.alloc_steps_[0][2].item<int>(),
    ctx.alloc_steps_[0][3].item<int>(),
    0,
    *root->prev_allocs,
    root->current_allocs->unsqueeze(0),
    root->core_caps->unsqueeze(0)
  );
  root->policy = pol[0].cpu();
  root->value_sum = val[0].item<float>();
  return root;
}


auto TreeSearch::expand_node(const OptCtx &ctx, std::shared_ptr<Node> node) const -> void {
  if (node->terminal)
    return;
  node->children.emplace(); // Initialize empty dict
  int qubit0 = ctx.alloc_steps_[node->alloc_step][1].item<int>();
  int qubit1 = ctx.alloc_steps_[node->alloc_step][2].item<int>();
  int remaining_gates = ctx.alloc_steps_[node->alloc_step][3].item<int>();

  // The prev to terminal node has no next step, but it does have children which
  // contains the cost of each of the actions that can be taken from it
  bool pre_terminal = (node->alloc_step == (ctx.n_steps_ - 1));
  int slice_idx_children;
  if (not pre_terminal)
    slice_idx_children = ctx.alloc_steps_[node->alloc_step+1][0].item<int>();
  
  std::vector<at::Tensor> curr_allocs_v;
  std::vector<at::Tensor> core_caps_v;
  std::vector<std::shared_ptr<Node>> children_v;
  
  for (size_t action = 0; action < n_cores_; action++) {
    if ((*node->policy)[action].item<float>() < 1e-5)
      continue;
    float cost = action_cost(ctx, node, action);
    auto child = std::make_shared<Node>();
    (*node->children)[action] = {child, cost};
    if (pre_terminal) {
      child->terminal = true;
      continue;
    }
        
    child->alloc_step = node->alloc_step + 1;
    child->slice_idx = slice_idx_children;
    if (child->slice_idx != node->slice_idx) {
      child->current_allocs = n_cores_ * at::ones_like(*node->current_allocs).to(device_);
      child->prev_allocs = node->current_allocs->clone();
      (*child->prev_allocs)[qubit0] = action;
      if (qubit1 != -1)
        (*child->prev_allocs)[qubit1] = action;
      child->core_caps = ctx.core_caps_;
    } else {
      child->current_allocs = node->current_allocs->clone();
      (*child->current_allocs)[qubit0] = action;
      if (qubit1 != -1)
        (*child->current_allocs)[qubit1] = action;
      child->prev_allocs = node->prev_allocs;
      child->core_caps = node->core_caps->clone();
      (*child->core_caps)[action] -= (qubit1 == -1 ? 1 : 2);
      assert((*child->core_caps)[action].item<int>() >= 0); // Not enough space in core to expand
    }

    curr_allocs_v.push_back(*child->current_allocs);
    core_caps_v.push_back(*child->core_caps);
    children_v.push_back(child);
  }

  if (not children_v.empty()) {
    auto [pols, vals] = new_policy_and_val(
      ctx,
      qubit0,
      qubit1,
      remaining_gates,
      slice_idx_children,
      *children_v[0]->prev_allocs, // All children have the same prev_allocs
      at::stack(curr_allocs_v, 0),
      at::stack(core_caps_v, 0)
    );
    for (int i = 0; i < children_v.size(); i++) {
      children_v[i]->policy = pols[i];
      children_v[i]->value_sum = vals[i].item<float>();
    }
  }
}


auto TreeSearch::action_cost(
  const OptCtx &ctx,
  std::shared_ptr<const Node> node,
  int action
) const -> float {
  if (node->slice_idx == 0)
    return 0;

  auto qubit_move_cost = [&](int q) -> float {
    int prev_core = (*node->prev_allocs)[q].item<int>();
    return ctx.core_conns_[action][prev_core].item<float>();
  };

  int qubit0 = ctx.alloc_steps_[node->alloc_step][1].item<int>();
  int qubit1 = ctx.alloc_steps_[node->alloc_step][2].item<int>();

  if (qubit1 != -1)
    return qubit_move_cost(qubit0) + qubit_move_cost(qubit1);
  return qubit_move_cost(qubit0);
}


auto TreeSearch::backprop(
  std::vector<std::tuple<std::shared_ptr<Node>, int>>& search_path,
  float discount_factor
) const -> void {
  // Terminal nodes need to be updated because at the last alloc we need to know visit count
  if (std::get<0>(search_path.back())->terminal)
    std::get<0>(search_path.back())->visit_count += 1;

  for (int i = search_path.size() - 2; i >= 0; i--) {
    auto node = std::get<0>(search_path[i]);
    auto next_node = std::get<0>(search_path[i+1]);
    int action = std::get<1>(search_path[i+1]);
    float action_cost = node->action_cost(action);
    node->value_sum +=  action_cost + (1 - discount_factor) * next_node->value();
    node->visit_count += 1;
  }
}


auto TreeSearch::ucb(
  const OptCtx &ctx,
  std::shared_ptr<const Node> node,
  int action
) const -> float {
  float rem_gates = ctx.alloc_steps_[node->alloc_step][3].item<float>();
  float q_v = (node->get_child(action)->value() + node->action_cost(action)) / (rem_gates+1);
  float q_inv = 2.0 / (1.0 + q_v) - 1;
  float prob_a = (*node->policy)[action].item<float>();
  float vc = node->visit_count;
  float vc_act = node->get_child(action)->visit_count;
  float uncert = prob_a * std::sqrt(vc) / (1 + vc_act)
                  * (ctx.cfg_.ucb_c1 + std::log(vc + ctx.cfg_.ucb_c2 + 1)/ctx.cfg_.ucb_c2);
  return q_inv + uncert;
}


auto TreeSearch::select_child(
  const OptCtx &ctx,
  std::shared_ptr<const Node> current_node
) const -> std::tuple<std::shared_ptr<Node>, int> {
  double max_ucb = -std::numeric_limits<double>::infinity();
  int best_action = -1;

  for (const auto& [action, _] : *current_node->children) {
    double ucb_value = ucb(ctx, current_node, action);
    assert(not std::isnan(ucb_value));
    assert(not std::isinf(ucb_value));
    if (ucb_value > max_ucb) {
      max_ucb = ucb_value;
      best_action = action;
    }
  }
  assert(best_action != -1);
  return {current_node->get_child(best_action), best_action};
}


auto TreeSearch::exploration_ratio(const OptCtx &ctx, int n_exp_nodes) const -> float {
  float theoretical_n_exp_nodes =
    ctx.n_steps_ * ctx.cfg_.target_tree_size * (n_cores_ - 1)/n_cores_;
  return n_exp_nodes/theoretical_n_exp_nodes;
}