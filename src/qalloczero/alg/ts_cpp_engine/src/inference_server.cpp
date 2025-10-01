#include "inference_server.hpp"
#include <torch/extension.h>
#include <torch/script.h>



auto InferenceServer::add_model(const std::string& path_to_pth, at::Device device) -> void {
  if (not models_.empty()) {
    throw std::runtime_error("Tried to change a model while there are execution contexts active");
  }
  try {
    auto model_unopt = torch::jit::load(path_to_pth);
    model_unopt.to(device);
    model_unopt.eval();
    model_ = torch::jit::optimize_for_inference(model_unopt);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load model: " + std::string(e.what()));
  }
}

auto InferenceServer::rm_model() -> void {
  if (not models_.empty()) {
    throw std::runtime_error("Tried to remove a model while there are execution contexts active");
  }
  model_ = std::nullopt;
}

auto InferenceServer::has_model() const -> bool {
  return model_.has_value();
}


auto InferenceServer::new_context(size_t id) const -> void {
  // models_[id] = model_->clone();
}

auto InferenceServer::rm_context(size_t id) const -> void {
  // models_.erase(id);
}