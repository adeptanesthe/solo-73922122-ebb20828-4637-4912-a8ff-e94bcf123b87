"""Test configuration: tiny matrices are faster single-threaded on CPU."""

import torch

torch.set_num_threads(1)
