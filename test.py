import torch
import numpy as np
from matplotlib import pyplot as plt
# a= torch.Tensor([[0,0,1,2,0],[1,1,2,0,0]])
# a.requires_grad = True
# b = a == 2
# b =b.float()
# print(b.requires_grad)
#
# depth_mask = torch.Tensor([[0,0,0,1,0],[0,1,1,0,0]])
# final_mask = torch.mul(b,depth_mask)
# final_mask = 1-final_mask
# pred = torch.Tensor([[[0],[0],[0]],[[1],[4],[0]]])
# pred = pred.unsqueeze(dim=0)
# target = torch.Tensor([[0],[2]]).long()
# target = target.unsqueeze(dim=0)
# soft = torch.nn.CrossEntropyLoss(reduction='none')
# # print(soft(pred,target))
#
# d = torch.Tensor([[0,0,1,2,0],[1,1,2,0,0]])
# one = d == 1
# two = d == 2
# three = one | two
#
# print(three)

# plt.plot(bins,n)
# plt.show()
torch.manual_seed(100)
a = torch.randint(low=0,high=2,size=(4,512,512)).float()
b = torch.randint(low=0,high=2,size=(4,512,512)).float()
c = a*b
print(torch.mean(c))