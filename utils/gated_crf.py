import torch
import numpy as np
import torch.nn as nn
import torch

class GatedCRFLoss(nn.Module):

    def __init__(self, num_classes, image_shape, cuda=True, span=11):
        """
        num_classes: number of classes in the task
        image_shape: (batch, C, H, W)

        2 access points to this class
        constructor, gatedCRFLoss
        """
        super(GatedCRFLoss, self).__init__()
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.unfold = nn.Unfold(kernel_size=(2 * span) + 1, padding=span)
        self.softmax = nn.Softmax(dim=1)

        # construct the class compatability matrix in the constructore
        C = num_classes
        M = torch.ones(C, C).float()
        temp = torch.ones(C)
        diag = torch.diag(temp).float()
        self.M = torch.flatten(M - diag).to(self.device)
        # construct the index tensor
        self.index_tensor = torch.as_tensor(self.generate_index_matrix(image_shape), dtype=torch.float,device=self.device)

    def generate_kernel_matrix(self, image_batch, y_hat, index_tensor, span=11, sig_rgb=0.1, sig_xy=6):
        """
        image_batch: 4 dimensional tensor of the form (batch, channels, H, W)
        y_hat: 4 dimensional tensor of the form (batch, classes, H, w)
        index_tenor: 4 dimensional tensor of the form (batch, 1, H, W)
        return: result which is of the form (batch, 1, 2span+1, 2span+1, H, W)
        """
        result = torch.zeros(
            (image_batch.shape[0], 2 * span + 1, 2 * span + 1, image_batch.shape[2], image_batch.shape[3]),
            device=self.device)

        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                dx1, dx2 = self._get_ind(dx)
                dy1, dy2 = self._get_ind(dy)

                # generate rgb gaussian
                feat_t1_rgb = image_batch[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                # pred_t1 = y_hat[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                feat_t2_rgb = image_batch[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]
                # pred_t2 = y_hat[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

                # generate index based gaussian
                feat_t1_ind = index_tensor[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                feat_t2_ind = index_tensor[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

                diff_rgb = feat_t2_rgb - feat_t1_rgb / sig_rgb
                diff_ind = feat_t2_ind - feat_t1_ind / sig_xy

                diff_rgb_sq = diff_rgb * diff_rgb
                diff_ind_sq = diff_ind * diff_ind

                exp_diff_rgb = torch.exp(torch.sum(-0.5 * diff_rgb_sq, dim=1))
                exp_diff_xy = torch.exp(torch.sum(-0.5 * diff_ind_sq, dim=1))

                # del feat_t1_rgb

                result[:, dx + span, dy + span, dx2:self._negative(dx1),
                dy2:self._negative(dy1)] = exp_diff_rgb + exp_diff_xy

        result_viewed = result.view(
            (image_batch.shape[0], 1, 2 * span + 1, 2 * span + 1, image_batch.shape[2], image_batch.shape[3]))
        del result
        return result_viewed

    @staticmethod
    def _get_ind(dz):
        if dz == 0:
            return 0, 0
        elif dz > 0:
            return dz, 0
        elif dz < 0:
            return 0, -dz

    @staticmethod
    def _negative(dz):
        if dz == 0:
            return None
        else:
            return -dz

    @staticmethod
    def generate_index_matrix(a):
        """
        a: shape of the matrix
        generates an index matrix for a 2 dimensional numpy matrix
        """
        result = np.zeros((a[0], 2, a[2], a[3]))
        for i in range(a[2]):
            for j in range(a[3]):
                result[:, 0, i, j] = i
                result[:, 1, i, j] = j
        return result


    def generate_source_tensor(self, image_batch_mask, span=11):
        """
        image_batch_mask: a batch-wise image mask for valid energy source, (batch, 1, H, W)
        return: (batch, 2*span+1, 2*span+1, H, W)
        """

        m_src = self.unfold(image_batch_mask)
        return m_src

    @staticmethod
    def generate_destination_tensor(image_batch_mask):
        '''
        image_batch_mask: a batch-wise image mask for valid energy destinations, (batch, 1, H, W)
        return: A tensor reshaping of the input tensor of size (batch, 1,1,1,H,W)
        '''
        temp = torch.unsqueeze(image_batch_mask, 2)
        result = torch.unsqueeze(temp, 2)
        return result

    def unfold_prediction(self, y_hat, span=11):
        '''
        y_hat: torch tensor of the form (batch, C, H, W)
        '''
        U = self.unfold(y_hat)
        return U

    def gatedCRFLoss(self, logit, target, image, weight=None, source_map=None, destination_map=None, lam=0.15):
        '''
        Implement the Gated CRF Loss for semi supervised segmentation tasks
        '''
        y_hat = self.softmax(logit)
        m_dst = 1 - destination_map
        L_gcrf = self.compute_gcrf_new(image, y_hat, source_map, m_dst, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
        criterion = criterion.to(self.device)
        L_ce = criterion(logit, target.long())
        L_ce = L_ce * destination_map
        L_ce = L_ce.mean()

        fin_loss = L_ce + lam * L_gcrf
        return fin_loss

    def compute_gcrf_new(self, image_batch, y_hat, m_src, m_dst, span=11, sig_rgb=0.1, sig_xy=6, device=None):
        '''
        image_batch: 4 dimensional tensor of the form (batch, channels, H, W)
        y_hat: 4 dimensional tensor of the form (batch, classes,  H, W)
        M: class compatibility matrix, a matrix of dimensions (classes, classes), with all ones and the principle diagonal elements being 0
        index_tensor: 4 dimensional tensor of the form (batch, 1, H, W)
        m_src: a mask of valid source pixels, {valid=1, invalid:0}
        m_dst: a mask of valid destination pixels, all annotated pixels are 0
        return: result which is of the form (batch, 1, 2span+1, 2span+1, H, W)
        '''
        M = self.M
        M = M.view(1, M.shape[0], 1, 1)
        result = torch.as_tensor(np.zeros((image_batch.shape[0], 2 * span + 1, 2 * span + 1)), dtype=torch.float,device=device)
        debug_count = 0
        for dx in range(-span, span + 1):

            # avoiding self labelling
            if dx == 0:
                continue
            for dy in range(-span, span + 1):

                # avoiding self labelling
                if dy == 0:
                    continue

                # retrieving indices for manipulation
                dx1, dx2 = self._get_ind(dx)
                dy1, dy2 = self._get_ind(dy)

                # cross compatibility computation on the prediction
                pred_t1 = y_hat[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                pred_t2 = y_hat[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]
                r = pred_t1.contiguous().view(-1, 3, 1).bmm(pred_t2.contiguous().view(-1, 1, 3))
                r = r.view(pred_t1.shape[0], 9, pred_t1.shape[2], pred_t1.shape[3])
                r = r * M

                # modify extract the corresponding regions from the source and destination maps
                m_src_mod = m_src[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]
                m_dst_mod = m_dst[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

                # generate rgb gaussian
                feat_t1_rgb = image_batch[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                feat_t2_rgb = image_batch[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

                # generate index based gaussian
                feat_t1_ind = self.index_tensor[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                feat_t2_ind = self.index_tensor[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

                diff_rgb = (feat_t2_rgb - feat_t1_rgb) / sig_rgb
                diff_ind = (feat_t2_ind - feat_t1_ind) / sig_xy

                diff_rgb_sq = diff_rgb * diff_rgb
                diff_ind_sq = diff_ind * diff_ind

                exp_diff_rgb = torch.exp(torch.sum(-0.5 * diff_rgb_sq, dim=1))
                exp_diff_xy = torch.exp(torch.sum(-0.5 * diff_ind_sq, dim=1))
                kernel_aggregate = exp_diff_rgb + exp_diff_xy

                # estimation of energy for span
                # estimate the kernel aggregate
                kernel_aggregate = torch.unsqueeze(kernel_aggregate, 1)

                # apply the source and destinations maps
                kernel_aggregate *= m_src_mod
                kernel_aggregate *= m_dst_mod

                pairwise_potential = kernel_aggregate * r
                pairwise_potential = torch.sum(pairwise_potential, dim=1)
                energy = torch.sum(pairwise_potential) / torch.sum(m_dst_mod)

                result[:, dx + span, dy + span] = energy
                debug_count += 1

                del r, pred_t1, pred_t2, feat_t1_rgb, feat_t2_rgb, feat_t1_ind, feat_t2_ind, diff_rgb, diff_ind, diff_rgb_sq, diff_ind_sq

        return result.mean()


if __name__ == "__main__":

    criterion = GatedCRFLoss(num_classes=3,image_shape=(16,3,512,512),cuda=True)
    for i in range(10):
        y_hat = torch.ones((16, 3, 512, 512)).cuda()
        image_batch = torch.ones((16, 3, 512, 512)).cuda()
        s_map = torch.ones((16, 1, 512, 512)).cuda()
        d_map = torch.randint(0, 2, (16, 1, 512, 512)).float().cuda()
        target = torch.ones((16, 512, 512)).cuda()
        loss = criterion.gatedCRFLoss(y_hat, target, image_batch, source_map=s_map, destination_map=d_map)
        print("Loss: ", loss)
        print("Loss value: ", loss.item())
        print("Max memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / 1e9))
