#include "inference_server.hpp"
#include <torch/extension.h>
#include <torch/script.h>



std::map<std::string, torch::jit::script::Module> InferenceServer::models{};

auto InferenceServer::add_model(std::string name, const std::string& path_to_pth) -> void {
  try {
    auto model_unopt = torch::jit::load(path_to_pth);
    model_unopt.eval();
    auto model = torch::jit::optimize_for_inference(model_unopt);
    model.eval();
    InferenceServer::models[std::move(name)] = std::move(model);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load model: " + std::string(e.what()));
  }
}

auto InferenceServer::rm_model(const std::string& name) -> void {
  auto elm = InferenceServer::models.find(name);
  if (elm != InferenceServer::models.end()) {
    InferenceServer::models.erase(elm);
  }
}


auto InferenceServer::has_model(const std::string& name) -> bool {
  return InferenceServer::models.find(name) != InferenceServer::models.end();
}