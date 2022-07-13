import torch

if torch.device=='mps':
    print("yes")
else:
    print("NO")