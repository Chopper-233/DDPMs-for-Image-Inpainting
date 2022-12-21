import torch
from torch import Tensor

mean1, logvar1, mean2, logvar2 = Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])

t = None
for obj in (mean1, logvar1, mean2, logvar2):
    if isinstance(obj, Tensor):
        t = obj
        break

logvar1, logvar2 = [x if isinstance(x, Tensor) else torch.tensor(x).to(t) for x in (logvar1, logvar2)]