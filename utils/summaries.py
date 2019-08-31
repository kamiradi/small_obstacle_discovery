import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
from dataloaders.utils import decode_seg_map_sequence,decode_segmap
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


    def vis_grid(self,writer,dataset,image,depth,target,pred,global_step,split):

        image = image.squeeze()
        image = image.astype(np.uint8)
        image = np.moveaxis(image,0,2)
        depth = depth.squeeze()
        depth = (depth - np.min(depth))/(np.max(depth)-np.min(depth))
        target = target.squeeze()
        target = decode_segmap(target,dataset=dataset)
        pred = pred.squeeze()
        pred = decode_segmap(pred,dataset=dataset).squeeze()

        fig = plt.figure(figsize=(7,20),dpi=150)
        ax1 = fig.add_subplot(411)
        ax1.imshow(image)
        ax2 = fig.add_subplot(412)
        ax2.imshow(pred)
        ax3 = fig.add_subplot(413)
        ax3.imshow(target)
        ax4 = fig.add_subplot(414)
        ax4.imshow(target)
        x,y = np.where(depth!=0)
        ax4.scatter(y,x,c='y',s=1)
        writer.add_image(split, figure_to_image(fig), global_step)
        plt.clf()

    def vis_depth(self,depth):
        norm_depth = [(depth[i]-np.min(depth[i]))/(np.max(depth[i])-np.min(depth[i])) for i in range(depth.shape[0])]
        norm_depth = np.array(norm_depth)
        norm_depth = plt.cm.plasma(norm_depth)
        norm_depth = np.moveaxis(norm_depth,4,1)
        norm_depth = norm_depth.squeeze()
        norm_depth = norm_depth[:,:3,:,:]
        norm_depth = torch.Tensor(norm_depth)
        #TODO: earlier visualisation showed small obs, not visible now. why?
        return norm_depth
