import torch
import sys
sys.path.append('/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/')
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder, build_decoder_depth
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
        def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                                 sync_bn=True, freeze_bn=False, depth=False):
                super(DeepLab, self).__init__()
                if backbone == 'drn':
                        output_stride = 8

                if sync_bn == True:
                        BatchNorm = SynchronizedBatchNorm2d
                else:
                        BatchNorm = nn.BatchNorm2d

                self.backbone = build_backbone(backbone, output_stride,
                                               BatchNorm, depth)
                self.aspp = build_aspp(backbone, output_stride, BatchNorm)
                self.decoder = build_decoder_depth(num_classes, backbone, BatchNorm)

                if freeze_bn:
                        self.freeze_bn()

        def forward(self, input):
                x, low_level_feat = self.backbone(input)
                x = self.aspp(x)
                # change related to uncertainty
                x, conf = self.decoder(x, low_level_feat)
                x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

                # change related to uncertainty
                conf = F.interpolate(conf, size=input.size()[2:],
                                     mode='bilinear', align_corners=True)

                return x, conf

        def freeze_bn(self):
                for m in self.modules():
                        if isinstance(m, SynchronizedBatchNorm2d):
                                m.eval()
                        elif isinstance(m, nn.BatchNorm2d):
                                m.eval()

        def get_1x_lr_params(self):
                modules = [self.backbone]
                for i in range(len(modules)):
                        for m in modules[i].named_modules():
                                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                                or isinstance(m[1], nn.BatchNorm2d):
                                        for p in m[1].parameters():
                                                if p.requires_grad:
                                                        yield p

        def get_10x_lr_params(self):
                modules = [self.aspp, self.decoder]
                for i in range(len(modules)):
                        for m in modules[i].named_modules():
                                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                                or isinstance(m[1], nn.BatchNorm2d):
                                        for p in m[1].parameters():
                                                if p.requires_grad:
                                                        yield p

class DeepLab_depth(nn.Module):
        def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                                 sync_bn=True, freeze_bn=False, depth=True):
                super(DeepLab_depth, self).__init__()
                if backbone == 'drn':
                        output_stride = 8

                if sync_bn == True:
                        BatchNorm = SynchronizedBatchNorm2d
                else:
                        BatchNorm = nn.BatchNorm2d

                self.backbone = build_backbone(backbone, output_stride,
                                               BatchNorm, depth)
                self.aspp = build_aspp(backbone, output_stride, BatchNorm)
                self.decoder = build_decoder_depth(num_classes, backbone, BatchNorm)

                if freeze_bn:
                        self.freeze_bn()

        def forward(self, input):
                x, low_level_feat = self.backbone(input)
                x = self.aspp(x)
                # change related to uncertainty
                x, conf = self.decoder(x, low_level_feat)
                x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

                # change related to uncertainty
                conf = F.interpolate(conf, size=input.size()[2:],
                                     mode='bilinear', align_corners=True)

                return x, conf

        def freeze_bn(self):
                for m in self.modules():
                        if isinstance(m, SynchronizedBatchNorm2d):
                                m.eval()
                        elif isinstance(m, nn.BatchNorm2d):
                                m.eval()

        def get_1x_lr_params(self):
                modules = [self.backbone]
                for i in range(len(modules)):
                        for m in modules[i].named_modules():
                                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                                or isinstance(m[1], nn.BatchNorm2d):
                                        for p in m[1].parameters():
                                                if p.requires_grad:
                                                        yield p

        def get_10x_lr_params(self):
                modules = [self.aspp, self.decoder]
                for i in range(len(modules)):
                        for m in modules[i].named_modules():
                                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                                or isinstance(m[1], nn.BatchNorm2d):
                                        for p in m[1].parameters():
                                                if p.requires_grad:
                                                        yield p

class TwinDeepLab(nn.Module):
    def __init__(self, num_classes, backbone, output_stride,
                 depth, depth_path, image_path, sync_bn=True,
                 freeze_bn=False):
        super(TwinDeepLab, self).__init__()
        self.depth_path = depth_path
        self.image_path = image_path
        self.image = build_image(num_classes, backbone, output_stride, sync_bn,
                                freeze_bn, depth, self.image_path)
        self.depth = build_depth(num_classes, backbone, output_stride, sync_bn,
                                 freeze_bn, depth, self.depth_path)
    def forward(self, input):
        depth = input[:, 3, :, :]
        image = input[:, :3, :, :]
        depth = torch.reshape(depth, (depth.shape[0], 1, depth.shape[1],
                                      depth.shape[2]))
        x_depth, conf_depth = self.depth(depth)
        x_image, conf_image = self.image(image)
        conf_probs = torch.cat((conf_image, conf_depth), 1)
        conf_probs = F.softmax(conf_probs, dim=1)
        image_probs = conf_probs[:, 0, :, :]
        depth_probs = conf_probs[:, 1, :, :]
        image_probs = torch.reshape(image_probs, (image_probs.shape[0], 1,
                                                  image_probs.shape[1],
                                                  image_probs.shape[2]))
        depth_probs = torch.reshape(depth_probs, (depth_probs.shape[0], 1,
                                                  depth_probs.shape[1],
                                                  depth_probs.shape[1]))

        x = (x_depth*depth_probs)+(x_image*image_probs)
        return x, image_probs, depth_probs

    def get_1x_lr_params(self):
        modules = [self.image, self.depth]
        for module in modules:
            sub_modules = [module.backbone]
            for i in range(len(sub_modules)):
                for m in sub_modules[i].named_modules():
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1],
                                                                 SynchronizedBatchNorm2d) \
                       or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.image,
                   self.depth]
        for module in modules:
            sub_modules = [module.aspp, module.decoder]
            for i in range(len(sub_modules)):
                for m in sub_modules[i].named_modules():
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1],
                                                                 SynchronizedBatchNorm2d) \
                       or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

def build_image(num_classes, backbone, output_stride, sync_bn, freeze_bn,
                depth, image_path):
    model = DeepLab(backbone, output_stride, num_classes, sync_bn, freeze_bn)
    checkpoint = torch.load(image_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def build_depth(num_classes, backbone, output_stride, sync_bn, freeze_bn,
                depth, depth_path):
    model = DeepLab_depth(backbone, output_stride, num_classes, sync_bn, freeze_bn)
    checkpoint = torch.load(depth_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def init_weight_file():
    model = DeepLab(backbone='drn', output_stride=16,num_classes=3)
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    checkpoint=torch.load('/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-drn.pth.tar',map_location='cpu')
    weight_shape=[3,256,1,1]
    conf_weight_shape=[1,256,1,1]
    checkpoint['state_dict']['decoder.diverge_conv_pred.weight']=nn.init.kaiming_normal_(torch.empty(weight_shape))
    checkpoint['state_dict']['decoder.diverge_conv_pred.bias']=nn.init.constant_(torch.empty(weight_shape[0]),0)
    checkpoint['state_dict']['decoder.diverge_conv_conf.weight']=nn.init.kaiming_normal_(torch.empty(conf_weight_shape))
    checkpoint['state_dict']['decoder.diverge_conv_conf.bias']=nn.init.constant_(torch.empty(conf_weight_shape[0]),0)
    del checkpoint['state_dict']['decoder.last_conv.8.weight']
    del checkpoint['state_dict']['decoder.last_conv.8.bias']
    f=open("/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-image_input_heteroscedastic.pth","wb")
    torch.save(checkpoint,f)
    f.close()
    # for i,param in enumerate(checkpoint['state_dict']):
    #         print(param," shape", checkpoint['state_dict'][param].shape)
    # depth_weight_shape = [16,1,7,7]
    # depth_channel_weights = nn.init.kaiming_normal_(torch.empty(depth_weight_shape))
    # temp = torch.cat((checkpoint['state_dict']['backbone.layer0.0.weight'],
    #           depth_channel_weights), 1)
    # print(temp.shape)
    # checkpoint['state_dict']['backbone.layer0.0.weight'] = temp
    # checkpoint['state_dict']['backbone.layer0.0.weight'] = nn.init.kaiming_normal_(torch.empty(depth_weight_shape))
    # f=open("/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-depth_input_heteroscedastic.pth","wb")
    # torch.save(checkpoint,f)
    # f.close()

def twindeeplabtest():
    tensor = torch.rand([2, 4, 512, 512])
    image = '/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-image_input_heteroscedastic.pth'
    depth = '/home/aditya/small_obstacle_ws/Small_Obstacle_Segmentation/deeplab-small_obs-depth_input_heteroscedastic.pth'
    model = TwinDeepLab(backbone='drn', output_stride=16, num_classes=3,
                        depth_path=depth, image_path=image, depth=True)
    output, image_conf, depth_conf = model(tensor)
    print('output shape: ',output.shape)
    print('image probs: ', image_conf.shape)
if __name__ == "__main__":
    # init_weight_file()
    twindeeplabtest()
