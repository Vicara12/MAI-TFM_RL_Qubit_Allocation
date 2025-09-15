#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <optional>
#include "inference_server.hpp"

namespace py = pybind11;

class TSEngine {
public:
    TSEngine() {}

    auto load_model(const std::string &path) -> void {
        InferenceServer::add_model("pred_model", path);
    }


    auto test_fwd(
        at::Tensor& qubits,
        at::Tensor& prev_core_allocs,
        at::Tensor& current_core_allocs,
        at::Tensor& core_capacities,
        at::Tensor& circuit_emb,
        at::Tensor& slice_adj_mat
    ) -> pybind11::tuple {
        

        auto outputs = InferenceServer::pack_and_infer(
            "pred_model",
            qubits,
            prev_core_allocs,
            current_core_allocs,
            core_capacities,
            circuit_emb,
            slice_adj_mat
        );

        if (outputs.isTuple()) {
            auto tuple_elements = outputs.toTuple()->elements();
            py::tuple py_out(tuple_elements.size());
            for (size_t i = 0; i < tuple_elements.size(); ++i) {
                py_out[i] = tuple_elements[i].toTensor();
            }
            return py_out;
        }

        // Single output
        py::tuple py_out(1);
        py_out[0] = outputs.toTensor();
        return py_out;
    }

private:
    std::optional<torch::jit::script::Module> model_ = std::nullopt;
};

PYBIND11_MODULE(ts_cpp_engine, m) {
    pybind11::class_<TSEngine>(m, "TSEngine")
        .def(pybind11::init<>())
        .def("load_model", &TSEngine::load_model)
        .def("test_fwd", &TSEngine::test_fwd);
}