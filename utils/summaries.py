import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image
from dataloaders.utils import decode_seg_map_sequence, decode_confidence_map_sequence, decode_segmap
import numpy as np
from matplotlib import pyplot as plt

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

    # def visualize_image(self, writer, dataset, image, target, output,
    #                     global_step, flag='imviz'):
    #     grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
    #     writer.add_image(flag+'/Image', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image(flag+'/Predicted label', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image(flag+'/Groundtruth label', grid_image, global_step)
    def visualize_image(self, writer, dataset, image, target, output,
                        global_step,num_image=3, flag='imviz'):
        # grid_image = make_grid(decode_confidence_map_sequence(conf[:num_image].detach().cpu().numpy()), num_image, normalize=False, range=(0, 255))
        # writer.add_image(flag+'/Conf', grid_image, global_step)
        grid_image = make_grid(image[:num_image].clone().cpu().data, num_image, normalize=True)
        writer.add_image(flag+'/Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:num_image], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), num_image, normalize=False, range=(0, 255))
        writer.add_image(flag+'/Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:num_image], 1).detach().cpu().numpy(),
                                                       dataset=dataset), num_image, normalize=False, range=(0, 255))
        writer.add_image(flag+'/Groundtruth label', grid_image, global_step)

    def vis_grid(self,writer,dataset,image,target,pred,global_step,split):

        image = image.squeeze()
        image = image.astype(np.uint8)
        image = np.moveaxis(image,0,2)
        # depth = depth.squeeze()
        # depth = (depth - np.min(depth))/(np.max(depth)-np.min(depth))
        target = target.squeeze()
        seg_mask = target == 2
        target = decode_segmap(target,dataset=dataset)
        pred = pred.squeeze()
        pred = decode_segmap(pred,dataset=dataset).squeeze()

        fig = plt.figure(figsize=(7,25),dpi=150)
        ax1 = fig.add_subplot(411)
        ax1.imshow(image)
        ax2 = fig.add_subplot(412)
        ax2.imshow(pred)
        ax3 = fig.add_subplot(413)
        ax3.imshow(target)
        ax4 = fig.add_subplot(414)
        ax4.imshow(target)
        writer.add_image(split, figure_to_image(fig), global_step)
        plt.clf()
