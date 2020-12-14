import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

def createBasicConv(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, bias=False, **kwargs),
        nn.ReLU(inplace=True)
        )

def createNormConv(in_chs, out_chs, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs, bias=False, **kwargs),
        nn.BatchNorm1d(out_chs),
        nn.ReLU(inplace=True)
        )
