#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <optional>
#include "tree_search.hpp"
#include "inference_server.hpp"


class TSEngine {
    TreeSearch ts_;
    at::Device device_;

    at::Device device_from_string(const std::string& dev_str) {
        if (dev_str == "cpu") {
            return at::Device(torch::kCPU);
        } else if (dev_str == "cuda" || dev_str == "cuda:0") {
            return at::Device(torch::kCUDA, 0);  // default GPU 0
        } else if (dev_str.rfind("cuda:", 0) == 0) {  // starts with "cuda:"
            int idx = std::stoi(dev_str.substr(5));
            return at::Device(torch::kCUDA, idx);
        } else {
            throw std::runtime_error("Unknown device string: " + dev_str);
        }
    }

public:
    TSEngine(
        int n_qubits,
        const at::Tensor& core_caps,
        const at::Tensor& core_conns,
        bool verbose,
        const std::string &device)
    : device_(device_from_string(device))
    , ts_(n_qubits, core_caps, core_conns, verbose, device_from_string(device))
    {}

    auto load_model(const std::string &name, const std::string &path) -> void {
        InferenceServer::add_model(name, path, device_);
    }

    auto has_model(const std::string &name) -> bool {
        return InferenceServer::has_model(name);
    }

    auto rm_model(const std::string &name) -> void {
        return InferenceServer::rm_model(name);
    }

    auto optimize(
        const at::Tensor& slice_adjm,
        const at::Tensor& circuit_embs,
        const at::Tensor& alloc_steps,
        TreeSearch::OptConfig cfg,
        bool ret_train_data
    ) -> std::tuple<at::Tensor, int, float, std::optional<TreeSearch::TrainData>> {
        // TODO: add train data
        return ts_.optimize(
            slice_adjm,
            circuit_embs,
            alloc_steps,
            cfg,
            ret_train_data
        );
    }
};



PYBIND11_MODULE(ts_cpp_engine, m) {
    pybind11::class_<TreeSearch::OptConfig>(m, "TseOptConfig")
        .def(py::init<>())
        .def_readwrite("target_tree_size",  &TreeSearch::OptConfig::target_tree_size)
        .def_readwrite("noise",             &TreeSearch::OptConfig::noise)
        .def_readwrite("dirichlet_alpha",   &TreeSearch::OptConfig::dirichlet_alpha)
        .def_readwrite("discount_factor",   &TreeSearch::OptConfig::discount_factor)
        .def_readwrite("action_sel_temp",   &TreeSearch::OptConfig::action_sel_temp)
        .def_readwrite("ucb_c1",            &TreeSearch::OptConfig::ucb_c1)
        .def_readwrite("ucb_c2",            &TreeSearch::OptConfig::ucb_c2);

    pybind11::class_<TreeSearch::TrainData>(m, "TseTrainData")
        .def(py::init<int, int, int>())
        .def_readwrite("qubits",      &TreeSearch::TrainData::qubits)
        .def_readwrite("prev_allocs", &TreeSearch::TrainData::prev_allocs)
        .def_readwrite("curr_allocs", &TreeSearch::TrainData::curr_allocs)
        .def_readwrite("core_caps",   &TreeSearch::TrainData::core_caps)
        .def_readwrite("slice_idx",   &TreeSearch::TrainData::slice_idx)
        .def_readwrite("logits",      &TreeSearch::TrainData::logits)
        .def_readwrite("value",       &TreeSearch::TrainData::value);
    
    pybind11::class_<TSEngine>(m, "TSEngine")
        .def(py::init<int, const at::Tensor&, const at::Tensor&, bool, const std::string&>())
        .def("load_model", &TSEngine::load_model)
        .def("has_model", &TSEngine::has_model)
        .def("rm_model", &TSEngine::rm_model)
        .def("optimize", &TSEngine::optimize);
}