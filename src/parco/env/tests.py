from env import QubitAllocEnv  
from generator import CircuitGenerator

if __name__ == "__main__":    
    generator_params = dict(num_slices=10, num_qubits=8)
    env = QubitAllocEnv(generator_params=generator_params, num_cores=2, core_capacity=4)
    env.reset(batch_size=32)

