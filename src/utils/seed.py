from contextlib import contextmanager
import torch
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def temporary_seed(seed):
    # Save states
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    gpu_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    
    # Set temporary seed
    seed_everything(seed)
        
    try:
        yield
    finally:
        # Restore states
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        for i, state in enumerate(gpu_states):
            torch.cuda.set_rng_state(state, i)
