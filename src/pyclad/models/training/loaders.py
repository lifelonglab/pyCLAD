import numpy as np
from torch import as_tensor
from torch.utils.data import DataLoader, TensorDataset


def float_tensor_loader(data: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap an array of samples in a DataLoader of float32 tensors.

    Shared by the torch strategies and the model adapter so the numpy->tensor->loader
    conversion lives in one place.
    """
    tensor = as_tensor(np.asarray(data, dtype=np.float32))
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)
