"""Keep video model tests from changing other modalities' RNG state."""

import pytest


@pytest.fixture(autouse=True)
def restore_random_state():
    """Restore the process-wide state modified while constructing and training test models."""
    import torch

    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
