#include <torch/extension.h>
#include <torch/script.h>
#include <vector>
#include <optional>

namespace py = pybind11;

class TSEngine {
public:
    TSEngine() {}

    auto load_model(const std::string &path) -> void {
        try {
            model_ = torch::jit::optimize_for_inference(torch::jit::load(path));
            model_->eval();
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model: " + std::string(e.what()));
        }
    }

    // The forward call expects IValues, not Tensors, so we need this function to cast them
    inline auto to_ivalue(const torch::Tensor& t) -> torch::jit::IValue {
        return t;
    }

    // This is a variadic template function to map any number of input arguments to the forward call
    template <typename... Args>
    auto forward(Args&&... args) -> torch::jit::IValue {
        torch::NoGradGuard no_grad;
        if (model_) {
            std::vector<torch::jit::IValue> inputs = { to_ivalue(std::forward<Args>(args))... };
            return model_->forward(inputs);
        } else {
            throw std::runtime_error("Failed to call forward: no model loaded");
        }
    }

    auto test_fwd(
        at::Tensor& qubits,
        at::Tensor& prev_core_allocs,
        at::Tensor& current_core_allocs,
        at::Tensor& core_capacities,
        at::Tensor& circuit_emb,
        at::Tensor& slice_adj_mat
    ) -> pybind11::tuple {
        

        auto outputs = forward(
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