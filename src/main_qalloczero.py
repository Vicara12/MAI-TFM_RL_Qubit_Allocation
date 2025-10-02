from qalloczero.scripts.test_alphazero import (
  testing_circuit_enc,
  testing_pred_model,
  test_cpp_engine,
  test_alphazero,
)
from qalloczero.scripts.parameter_finetune import (
  linear_search,
)
from qalloczero.scripts.test_directalloc import (
  test_direct_alloc
)



if __name__ == "__main__":
  # testing_circuit_enc()
  # testing_pred_model()
  # test_cpp_engine()
  # grid_search()
  # test_alphazero()
  # linear_search()
  test_direct_alloc()