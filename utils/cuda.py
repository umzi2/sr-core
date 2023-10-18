import torch


def safe_cuda_cache_empty():
    """
    Empties the CUDA cache if CUDA is available. Hopefully without causing any errors.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
