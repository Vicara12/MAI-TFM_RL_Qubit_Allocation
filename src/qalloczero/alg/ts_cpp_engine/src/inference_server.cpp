#include "inference_server.hpp"
#include <torch/extension.h>
#include <torch/script.h>



auto InferenceServer::add_model(std::string name, const std::string& path_to_pth) -> void {
  try {
    auto model = torch::jit::optimize_for_inference(torch::jit::load(path_to_pth));
    model.eval();
    InferenceServer::models[name] = std::move(model);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load model: " + std::string(e.what()));
  }
}


auto InferenceServer::has_model(const std::string& name) -> bool {
  return InferenceServer::models.find(name) == InferenceServer::models.end();
}


template <typename... Args>
auto infer(const std::string& name, Args&&... args) -> std::vector<torch::jit::IValue> {
    torch::NoGradGuard no_grad;
    auto model = InferenceServer::models[name];
    std::vector<torch::jit::IValue> inputs = {
      InferenceServer::to_ivalue(std::forward<Args>(args))...
    };
    auto output model->forward(inputs);

    if (outputs.isTuple()) {
      auto tuple_elements = outputs.toTuple()->elements();
      std::vector<torch::jit::IValue> out_vec(tuple_elements.size());
      for (size_t i = 0; i < tuple_elements.size(); ++i) {
        out_vec[i] = tuple_elements[i].toTensor();
      }
      return out_vec;
    }

    return {output};
}


template <typename... Args>
static auto pack_and_infer(const std::string& name, Args&&... args) -> torch::jit::IValue {
    return infer(name, args.unsqueeze(0)...);
}