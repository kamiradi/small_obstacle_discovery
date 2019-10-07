from termcolor import colored
import torch
import torch.nn as nn
import gc
import torch
import numpy as np
import torch.nn as nn
from fastai.utils.mem import GPUMemTrace, gpu_mem_trace, gpu_mem_get
from tqdm import tqdm_notebook


class GatedCRFLoss(nn.Module):
    # @gpu_mem_trace
    def __init__(self, num_classes, image_shape, cuda=True, span=11):
        # self.memtrace = GPUMemTrace()
        """
        num_classes: number of classes in the task
        image_shape: (batch, C, H, W)

        2 access points to this class
        constructor, gatedCRFLoss
        """
        super(GatedCRFLoss, self).__init__() 
        self.device = torch.device('cuda') if cuda else torch.device('cpu')

        self.unfold = nn.Unfold(kernel_size=(2*span) + 1, padding=span)
        self.softmax = nn.Softmax(dim=1)

        # construct the class compatability matrix in the constructore
        C = num_classes
        M = torch.ones(C, C).float()
        temp = torch.ones(C)
        diag = torch.diag(temp).float()
        self.M = torch.flatten(M - diag).to(self.device)
        # construct the index tensor
        self.index_tensor = torch.as_tensor(self.generate_index_matrix(image_shape),dtype=torch.float,device=self.device)

    # @gpu_mem_trace
    def generate_kernel_matrix(self, image_batch, y_hat, index_tensor, span=11, sig_rgb=0.1, sig_xy=6):
        """
        image_batch: 4 dimensional tensor of the form (batch, channels, H, W)
        y_hat: 4 dimensional tensor of the form (batch, classes, H, w)
        index_tenor: 4 dimensional tensor of the form (batch, 1, H, W)
        return: result which is of the form (batch, 1, 2span+1, 2span+1, H, W)
        """
        # mtrace = GPUMemTrace(ctx='aggregate kernel generation')
        result = torch.zeros((image_batch.shape[0], 2 * span + 1, 2 * span + 1, image_batch.shape[2], image_batch.shape[3]),device=self.device)

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
            

                result[:, dx + span, dy + span, dx2:self._negative(dx1),dy2:self._negative(dy1)] = exp_diff_rgb + exp_diff_xy


        # mtrace.report('after for loop')
        result_viewed = result.view((image_batch.shape[0], 1, 2 * span + 1, 2 * span + 1, image_batch.shape[2], image_batch.shape[3]))
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

    # @gpu_mem_trace
    def generate_source_tensor(self, image_batch_mask, span=11):
        """
        image_batch_mask: a batch-wise image mask for valid energy source, (batch, 1, H, W)
        return: (batch, 2*span+1, 2*span+1, H, W)
        """
        
        m_src = self.unfold(image_batch_mask)
        print('m_src tensor device', m_src.device)
        # unfold = nn.Unfold(kernel_size=2 * span + 1, padding=span)
        # m_src = unfold(image_batch_mask)
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

    # @gpu_mem_trace
    def unfold_prediction(self, y_hat, span=11):
        '''
        y_hat: torch tensor of the form (batch, C, H, W)
        '''
        U = self.unfold(y_hat)
        print('unfolded prediction tensor device', U.device)
        # unfold = nn.Unfold(kernel_size=2 * span + 1, padding=span) 
        # U = unfold(y_hat)
        return U

    # @gpu_mem_trace
    def gatedCRFLoss(self, logit, target, image, weight=None, source_map=None, destination_map=None, lam=0.15):
        '''
        Implement the Gated CRF Loss for semi supervised segmentation tasks
        '''
        y_hat = self.softmax(logit) 
        m_dst = 1 - destination_map
        L_gcrf = self.compute_gcrf_new(image.data, y_hat.data, source_map.data, m_dst.data, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight,reduction='none')
        criterion = criterion.to(self.device)
        L_ce = criterion(logit, target.long())
        L_ce = L_ce * destination_map
        L_ce = L_ce.mean()

        fin_loss = L_ce + lam * L_gcrf
        return fin_loss

    # @gpu_mem_trace
    def compute_gcrf(self, y_hat, image, source_map=None, destination_map=None, use_dest=True):
        '''
        Interface info
        y_hat : (Batch, C, H, W) tensor of predictions
        image: (Batch, C, H, W) tensor of a batch of images
        source_map: (Batch, 1, H, W) tensor of valid energy source positions
        destination_map: (Batch, 1, H, W) tensor of valid energy destination positions
        '''
        mtrace = GPUMemTrace(ctx='computation of gcrf')
        # unfolding the prediction tensor into the energy sub-regions
        U = self.unfold_prediction(y_hat.to('cpu'))
        mtrace.report('prediction unfold')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        # initialising a few tensors to calculate aggregate kernels
        m_src = self.generate_source_tensor(source_map.to('cpu'))
        mtrace.report('source tensor unfold')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        m_dst = self.generate_destination_tensor(destination_map.to('cpu'))
        mtrace.report('destination tensor unsaqueezed')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))

        # generating aggregate kernels
        k = self.generate_kernel_matrix(image, self.index_tensor)
        k_shape = k.shape
        mtrace.report('kernel matrix created : extensive for loops used')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))

        k_viewed = k.view(m_src.shape[0], m_src.shape[1], m_src.shape[2])
        mtrace.report('kernel matrix.view()')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del k
        mtrace.report('k deleted')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        m_dst_reshaped = m_dst.view(m_src.shape[0], 1, m_src.shape[2])
        mtrace.report('destination matrix.view()')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        K = m_src.to('cuda') * k_viewed.to('cuda') # instance 1 of element wise multiplication
        mtrace.report('K = m_src*k_viewed')
        if use_dest:
            K = m_dst_reshaped.to('cuda') * K # instance 2 of element wise multiplication
        mtrace.report('K = m_dst*K')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del m_src
        del k_viewed
        mtrace.report('m_src and k_viewed deleted')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))

        # generating energy tensor
        K_viewed = K.to('cpu').view(k_shape[0], k_shape[1], k_shape[2], k_shape[3], k_shape[4], k_shape[5])
        mtrace.report('K matrix.view()')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del K
        mtrace.report('K deleted')
        # E = U.view(k_shape[0], 3, k_shape[2], k_shape[3], k_shape[4], k_shape[5]).to('cuda')*K_viewed.to('cuda')
        K_viewed = U.view(k_shape[0], 3, k_shape[2], k_shape[3], k_shape[4], k_shape[5]).to('cuda')*K_viewed.to('cuda')
        mtrace.report('unfolded prediction.view(), and energy estimation')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del U
        mtrace.report('U(prediction) deleted')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        # E = K_viewed.to('cuda') * U_viewed.to('cuda') # instance 3 of elementwise multiplication
        # mtrace.report('Energy estimated')
        # E_permuted = E.to('cpu').permute(0, 1, 4, 5, 2, 3)
        E_permuted = K_viewed.to('cpu').permute(0, 1, 4, 5, 2, 3)
        mtrace.report('Energy matrix permuted')
        # del E
        del K_viewed
        mtrace.report('E and K_viewed deleted')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        # del U_viewed
        # mtrace.report('Energy matrix permuted')
        E_viewed = E_permuted.contiguous().view(E_permuted.shape[0], E_permuted.shape[1],
                                                E_permuted.shape[2] * E_permuted.shape[3],
                                                E_permuted.shape[4] * E_permuted.shape[5])
        mtrace.report('Energy matrix.contiguous().view()')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))


        # Loss estimation
        J_e_viewed = torch.ones((E_permuted.shape[4] * E_permuted.shape[5]),dtype=torch.float, device='cpu')
        Egy = torch.matmul(E_viewed.to('cuda'), J_e_viewed.to('cuda'))
        mtrace.report('matmul(Energy, J of ones)')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del E_viewed
        del E_permuted
        del J_e_viewed

        y_hat_viewed = y_hat.view(y_hat.shape[0], y_hat.shape[1], y_hat.shape[2] * y_hat.shape[3])
        mtrace.report('prediction.view()')
        temp = torch.matmul(Egy, y_hat_viewed.permute(0, 2, 1))
        mtrace.report('matmul(energy, prediction')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del Egy
        del y_hat_viewed

        Egy_temp = self.M * temp
        mtrace.report('class compatability matrix multiplies')
        J_1_3 = torch.ones((1, 3),dtype=torch.float,device=self.device)
        J_3_1 = torch.ones((3, 1),dtype=torch.float,device=self.device)
        Egy_temp = torch.matmul(Egy_temp, J_3_1)
        Egy_final = torch.matmul(J_1_3, Egy_temp)
        mtrace.report('final energy calculated')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        del Egy_temp

        self.loss = Egy_final.mean() / m_dst.sum()
        del Egy_final
        mtrace.report('final energy divided by number of destination pixels')
        print("Memory allocated: {} GB".format(torch.cuda.max_memory_allocated()/1e9))
        # clear memory
        # gc.collect()

        return self.loss

    # @gpu_mem_trace
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
        loop_memory = 0.0
        M = M.view(1, M.shape[0], 1, 1)
        result = torch.as_tensor(np.zeros((2*span+1, 2*span+1)), dtype=torch.float, device=device)
        # mtrace.report('result tensor created')
        debug_count = 0
        for dx in range(-span, span+1):
            # avoiding self labelling
            if dx == 0:
                continue
            for dy in range(-span, span+1):
                # avoiding self labelling
                if dy == 0:
                    continue
                # retrieving indices for manipulation
                dx1, dx2 = self._get_ind(dx)
                dy1, dy2 = self._get_ind(dy)

                # cross compatibility computation on the prediction
                pred_t1 = y_hat[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                pred_t2 = y_hat[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

                # GPU mem profiling code

                pred_t1_contiguous = pred_t1.cpu().contiguous()
                pred_t2_contiguous = pred_t2.cpu().contiguous()

                pred_t1_c_v = pred_t1_contiguous.to(device).view(-1, 3, 1)
                pred_t2_c_v = pred_t2_contiguous.to(device).view(-1, 1, 3)

                r = pred_t1_c_v.bmm(pred_t2_c_v)
                del pred_t1_contiguous, pred_t2_contiguous

                r = r.view(pred_t1.shape[0], 9, pred_t1.shape[2], pred_t1.shape[3])
                r = r*M
    
                # modify extract the corresponding regions from the source and destination maps
                m_src_mod = m_src[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]
                m_dst_mod = m_dst[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]
    
                # generate rgb gaussian 
                feat_t1_rgb = image_batch[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                feat_t2_rgb = image_batch[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]

        
                # generate index based gaussian
                feat_t1_ind = self.index_tensor[:, :, dx1:self._negative(dx2), dy1:self._negative(dy2)]
                feat_t2_ind = self.index_tensor[:, :, dx2:self._negative(dx1), dy2:self._negative(dy1)]
                # mtrace.report('tensors extracted')
        
                diff_rgb = (feat_t2_rgb - feat_t1_rgb)/sig_rgb
                diff_ind = (feat_t2_ind - feat_t1_ind)/sig_xy
                # mtrace.report('differences computed')
        
                diff_rgb_sq = diff_rgb * diff_rgb
                diff_ind_sq = diff_ind * diff_ind
                # mtrace.report('squares of the diffs computed')
        
                exp_diff_rgb = torch.exp(torch.sum(-0.5 * diff_rgb_sq, dim=1))
                exp_diff_xy = torch.exp(torch.sum(-0.5 * diff_ind_sq, dim=1))
                # mtrace.report('individual kernels computed')

                kernel_aggregate = exp_diff_rgb + exp_diff_xy
                # mtrace.report('final kernel computed')
    
                # estimation of energy for span
                # estimate the kernel aggregate
                kernel_aggregate = torch.unsqueeze(kernel_aggregate, 1)
    
                # apply the source and destinations maps
                kernel_aggregate *= m_src_mod
                kernel_aggregate *= m_dst_mod
                # mtrace.report('gates applied')
                # print(colored('compute_gcrf_new: gates computed: {}'.format(gpu_mem_get(2)), 'yellow'))
    
                pairwise_potential = kernel_aggregate*r
                pairwise_potential= torch.sum(pairwise_potential, dim=1)
                energy = torch.sum(torch.unsqueeze(pairwise_potential, 1))/torch.sum(m_dst_mod)

                result[dx+span, dy+span] = energy
                debug_count += 1

                # deletions
                del r, pred_t1, pred_t2, feat_t1_rgb, feat_t2_rgb, feat_t1_ind, feat_t2_ind, diff_rgb, diff_ind, diff_rgb_sq, diff_ind_sq
                del pairwise_potential, kernel_aggregate, m_src_mod, m_dst_mod, exp_diff_rgb, exp_diff_xy

                # GPU mem profiling
                torch.cuda.empty_cache()
            # print(colored('compute_gcrf_new: stepping through x: {}' .format(gpu_mem_get(2)), 'blue'))
        return result.mean()

'''Segmentation losses prescribed by the default repository'''
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
        criterion = nn.CrossEntropyLoss(weight=weight,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def LidarCrossEntropyLoss(self, logit, target, lidar_bin_mask,
                              weight=None, class_value=2, lid_param=0.9):
        # lid_param = (torch.sum(lidar_bin_mask))/(lidar_bin_mask.shape[0]*lidar_bin_mask.shape[1]*lidar_bin_mask.shape[2])
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')

        if self.cuda:
            criterion= criterion.cuda()
        loss = criterion(logit, target.long())
        l1 = torch.mean(loss)
        l2 = torch.mean(loss*lidar_bin_mask)
        return ((1-lid_param)*l1 + lid_param*l2)

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




