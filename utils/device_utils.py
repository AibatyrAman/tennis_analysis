import torch

def get_best_device():
    """
    Selects the best available device for PyTorch execution.
    Prioritizes CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return 0 # Return device ID 0 for CUDA
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
