from __future__ import annotations
import torch

Tensor = torch.Tensor
tensor = torch.tensor
Module = torch.nn.Module
ModuleList = torch.nn.ModuleList
Parameter = torch.nn.Parameter
arange = torch.arange
concatenate = torch.cat
stack = torch.stack
linspace = torch.linspace
normalize = torch.nn.functional.normalize
pad = torch.nn.functional.pad
eye = torch.eye
einsum = torch.einsum
zeros = torch.zeros
zeros_like = torch.zeros_like
ones = torch.ones
ones_like = torch.ones_like
where = torch.where
complex = torch.complex
diag = torch.diag
softmax = torch.nn.functional.softmax
# constructors
as_tensor = torch.as_tensor

# random
rand = torch.rand
