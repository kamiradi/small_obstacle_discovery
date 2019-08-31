import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self,size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

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
        seg_mask = (target == 2) | (target == 1)                                      # Mask where specific class(small obstacle) is present
        seg_mask = seg_mask.float()
        final_mask = seg_mask*depth_mask
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

        complete_loss = l2*weighted_val + l1*(1-weighted_val)
        # complete_loss = l2*weighted_val + l1

        return complete_loss,l1,l2


    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




