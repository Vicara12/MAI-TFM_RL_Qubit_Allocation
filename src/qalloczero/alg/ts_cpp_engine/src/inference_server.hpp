#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>



class InferenceServer
{
  mutable std::optional<torch::jit::script::Module> model_;
  mutable std::map<size_t, torch::jit::script::Module&> models_;

  // The forward call expects IValues, not Tensors, so we need this function to cast them
  static inline auto to_ivalue(const torch::Tensor& t) -> torch::jit::IValue {return t;}

public:

  auto add_model(const std::string& path_to_pth, at::Device device) -> void;

  auto has_model() const -> bool;

  auto rm_model() -> void;

  auto new_context(size_t id) const -> void;

  auto rm_context(size_t id) const -> void;

  // Variadic template function to map any number of input arguments to the forward call
  template <typename... Args>
  auto infer(std::optional<size_t> ctx, Args&&... args) const -> std::vector<at::Tensor> {
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {
      InferenceServer::to_ivalue(std::forward<Args>(args))...
    };
    // auto outputs = (ctx.has_value() ? models_[*ctx].forward(inputs) : model_->forward(inputs));
    auto outputs = model_->forward(inputs);

    if (outputs.isTuple()) {
      auto tuple_elements = outputs.toTuple()->elements();
      std::vector<at::Tensor> out_vec(tuple_elements.size());
      for (size_t i = 0; i < tuple_elements.size(); ++i) {
        out_vec[i] = tuple_elements[i].toTensor();
      }
      return out_vec;
    }

    return std::vector<at::Tensor>{outputs.toTensor()};
  }
};