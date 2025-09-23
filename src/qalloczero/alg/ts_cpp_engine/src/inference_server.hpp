#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>



class InferenceServer
{
  static std::map<std::string, torch::jit::script::Module> models;

  // The forward call expects IValues, not Tensors, so we need this function to cast them
  static inline auto to_ivalue(const torch::Tensor& t) -> torch::jit::IValue {return t;}

public:

  static auto add_model(std::string name, const std::string& path_to_pth) -> void;

  static auto has_model(const std::string& name) -> bool;

  static auto rm_model(const std::string& name) -> void;

  // Variadic template function to map any number of input arguments to the forward call
  template <typename... Args>
  static auto infer(const std::string& name, Args&&... args) -> std::vector<at::Tensor>;

  // Infer method to call models that support batch with a single instance (unsqueeze all params)
  template <typename... Args>
  static auto pack_and_infer(const std::string& name, Args&&... args) -> std::vector<at::Tensor>;
};