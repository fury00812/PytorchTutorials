'''
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
PyTorchのウリであるautograd（自動微分）についてその使い方を紹介
'''
import torch

x = torch.ones(2,2,requires_grad=True)
print(x)
