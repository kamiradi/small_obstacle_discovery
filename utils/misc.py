import numpy as np
import torch

def get_index_tensor(shape):
    index_array = np.zeros((shape[0], 2, shape[2], shape[3]))
    dummy_array = np.ones((shape[2], shape[3]))
    x, y = np.where(dummy_array == 1)
    x, y = x.reshape(shape[2], shape[3]), y.reshape(shape[2], shape[3])
    index_array[:, 0] = x
    index_array[:, 1] = y
    return torch.from_numpy(index_array).float()