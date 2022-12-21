import functools
from typing import List

import torch
import torch.utils.data
import torchvision
from torchvision.utils import save_image as imsave
import torchvision.transforms as tvt
import os
from tqdm.auto import tqdm

from ddpm import DDPM
from unet import UNet
import torch
from torch import nn, Tensor
from torch.nn import Identity, Module, ModuleList, Linear, Sigmoid, GroupNorm, Conv2d, ConvTranspose2d, Dropout
from typing import Union, Tuple, List, Optional, Callable
import enum
from tqdm.auto import tqdm
from utils import normal_kl, mean_flat, discretized_gaussian_log_likelihood, gather, extract
import numpy as np
import torchvision as tv
import torchvision.transforms.functional as fn
from scipy import linalg

import math
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import Identity, Module, ModuleList, Linear, Sigmoid, GroupNorm, Conv2d, ConvTranspose2d, Dropout

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running test script")
print(dev)