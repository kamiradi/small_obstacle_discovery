import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SegmentationLosses(object):
    def __init__(self,size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        alpha = 0.2
        self.alpha = 0.25
        self.gamma = 2

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self,logit,target,weight):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=weight,size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss


    def LidarCrossEntropyLoss(self, logit, target, depth_mask, weight, weighted_val):
        seg_mask = (target == 2) | (target == 1)    # Mask where specific road and small obstacle is present
        # seg_mask = seg_mask.unsqueeze(dim=1)
        seg_mask = seg_mask.float()
        final_mask = seg_mask*depth_mask
        # final_mask = final_mask.squeeze(dim=1)
        neg_final_mask = 1-final_mask
        criterion = nn.CrossEntropyLoss(weight=weight,reduction='none')

        if self.cuda:
            criterion = criterion.cuda()

        l1 = criterion(logit,target.long())
        l1 = l1*neg_final_mask
        l1 = torch.mean(l1)
        l2 = criterion(logit,target.long())
        l2 = l2*final_mask
        l2 = torch.mean(l2)

        if weighted_val == "auto":
            count = torch.sum(final_mask)/(final_mask.shape[0]*final_mask.shape[1]*final_mask.shape[2]*final_mask.shape[3])
            complete_loss = l2*(1-count) + l1*count

        else:
            complete_loss = l2*weighted_val + l1*(1-weighted_val)

        return complete_loss,l1,l2


    def FocalLoss(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        F_loss = F_loss.cuda()
        print(F_loss.shape)
        return torch.mean(F_loss)


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




