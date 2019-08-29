import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
import numpy as np
from matplotlib import pyplot as plt

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image,depth,target, output, global_step,num_image=2):
        grid_image = make_grid(image[:num_image].clone().detach().cpu().data, num_image,normalize=True)
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(self.vis_depth(depth[:num_image].clone().detach().cpu().numpy()), num_image,normalize=True)
        writer.add_image('Depth_pred', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:num_image], 1)[1].detach().cpu().numpy(),dataset=dataset), num_image)
        writer.add_image('Predicted label', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:num_image], 1).detach().cpu().numpy(),dataset=dataset), num_image)
        writer.add_image('Groundtruth label', grid_image, global_step)

    def vis_depth(self,depth):
        norm_depth = [(depth[i]-np.min(depth[i]))/(np.max(depth[i])-np.min(depth[i])) for i in range(depth.shape[0])]
        norm_depth = np.array(norm_depth)
        norm_depth = plt.cm.plasma(norm_depth)
        norm_depth = np.moveaxis(norm_depth,4,1)
        norm_depth = norm_depth.squeeze()
        norm_depth = norm_depth[:,:3,:,:]
        norm_depth = torch.tensor(norm_depth)
        #TODO: earlier visualisation showed small obs, not visible now. why?
        return norm_depth