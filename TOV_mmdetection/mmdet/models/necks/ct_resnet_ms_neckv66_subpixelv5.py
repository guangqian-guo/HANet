# Copyright (c) OpenMMLab. All rights reserved.
from telnetlib import SE
from turtle import forward
from cv2 import norm
from matplotlib import scale
from matplotlib.pyplot import sca
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch
from tools.helper import to_2tuple, GELU
from timm.models.layers import DropPath
from ..builder import NECKS
from ..utils.feature_visiualization import show_fea
from .SEAttention import SEAttention
from .CBAM import CBAMBlock
@NECKS.register_module()
class CTResNetMSNeckv66_SPv5(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CTResNetMSNeckv66_SPv5, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()


        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        self.mask_layer4_1 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'),act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer4_2 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer4_3 = ConvModule(out_channels, 5, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=None, act_cfg=None)
        
        self.layer4_1_1 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'),)
        self.layer4_1_2 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer4_1_3 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer4_2_1 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer4_2_2 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer4_3 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer4_4 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        
        self.mask_layer3_1 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'),act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer3_2 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer3_3 = ConvModule(out_channels, 5, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=None, act_cfg=None)

        self.layer3_1_1 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer3_1_2 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer3_2 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer3_3 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer3_4 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))

        self.mask_layer2_1 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'),act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer2_2 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer2_3 = ConvModule(out_channels, 5, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=None, act_cfg=None)

        self.layer2_1 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer2_2 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer2_3 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer2_4 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        
        self.mask_layer1_1 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'),act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer1_2 = nn.Sequential(ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          ConvModule(out_channels, out_channels, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
                                          SEAttention(out_channels))
                                          
        self.mask_layer1_3 = ConvModule(out_channels, 5, 3, stride=1, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=None, act_cfg=None)

        self.layer1_1 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer1_2 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer1_3 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        self.layer1_4 = ConvModule(out_channels, out_channels, 1, padding=0, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))

        # add CBAM module
        self.cbam_layer1 = CBAMBlock(channel=64, reduction=4, kernel_size=3)
        self.cbam_layer2 = CBAMBlock(channel=64, reduction=4, kernel_size=3)
        self.cbam_layer3 = CBAMBlock(channel=64, reduction=4, kernel_size=3)
        self.cbam_layer4 = CBAMBlock(channel=64, reduction=4, kernel_size=3)

        # add upsample layer
        #----------------------------------------- layer 4 ----------------------------------------------------------------#
        self.upsample_layer_4 = LowLevel_FeaEnh(out_channels)
        self.upsample_layer_3 = LowLevel_FeaEnh(out_channels)
        self.upsample_layer_2 = LowLevel_FeaEnh(out_channels)
        self.upsample_layer_1 = LowLevel_FeaEnh(out_channels, is_firstlevel=True)


#----------------------------------------- layer 3 ----------------------------------------------------------------#
        self.lateral_layer3 = nn.Sequential(ConvModule(in_channels[-2]+in_channels[-3]+in_channels[-4], out_channels, 1,
                                        conv_cfg=dict(type='Conv2d'),act_cfg=None),
                                        ConvModule(out_channels, out_channels, 3, padding=1,
                                        conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN')),
                                        ConvModule(out_channels, out_channels, 1,
                                        conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN')))
        

#----------------------------------------- layer 2 ----------------------------------------------------------------#
       
        self.lateral_layer2_1 = nn.Sequential(ConvModule(64+in_channels[-3]+in_channels[-4], out_channels, 1,
                                        conv_cfg=dict(type='Conv2d'),act_cfg=None),
                                        ConvModule(out_channels, out_channels, 3, padding=1,
                                        conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN')),
                                        ConvModule(out_channels, out_channels, 1,
                                        conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN')))
        
#------------------------------------------ layer 1 -----------------------------------------------------------------#
        self.lateral_layer1_1 = nn.Sequential(ConvModule(64+in_channels[-4], out_channels, 1,
                                        conv_cfg=dict(type='Conv2d'),act_cfg=None),
                                        ConvModule(out_channels, out_channels, 3, padding=1,
                                        conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN')),
                                        ConvModule(out_channels, out_channels, 1,
                                        conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN')))
       
    



#---------------------------------------------------------------------------------------------------------#
    @auto_fp16()
    def forward(self, inputs,img_metas=None, gt_bboxes=None, gt_labels=None):
        """Forward function."""

        lowfeature = inputs[0]
        inputs = inputs[1:]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)

        # build mask layers
        # residual link
        self.mask4 = self.mask_layer4_1(laterals[-1]) + laterals[-1]
        self.mask4 = self.mask_layer4_2(self.mask4) + self.mask4
        self.mask4 = self.mask_layer4_3(self.mask4)
        
        self.mask3 = self.mask_layer3_1(laterals[-2]) + laterals[-2]
        self.mask3 = self.mask_layer3_2(self.mask3) + self.mask3
        self.mask3 = self.mask_layer3_3(self.mask3)

        self.mask2 = self.mask_layer2_1(laterals[-3]) + laterals[-3]
        self.mask2 = self.mask_layer2_2(self.mask2) + self.mask2
        self.mask2 = self.mask_layer2_3(self.mask2)

        self.mask1 = self.mask_layer1_1(laterals[-4]) + laterals[-4]
        self.mask1 = self.mask_layer1_2(self.mask1) + self.mask1
        self.mask1 = self.mask_layer1_3(self.mask1) 
        
        self.masks = [self.mask1, self.mask2, self.mask3, self.mask4]
        
        self.mask4 = F.softmax(self.mask4,dim=1)
        self.mask3 = F.softmax(self.mask3,dim=1)
        self.mask2 = F.softmax(self.mask2,dim=1)
        self.mask1 = F.softmax(self.mask1,dim=1)
      

        # build feature layers
        # layer 4
        layer4 = self.layer4_1_3(self.layer4_1_2(self.layer4_1_1(laterals[0] * (self.mask1[:,4,:,:]).unsqueeze(1)))) + \
                self.layer4_2_1(self.layer4_2_2(laterals[1] * (self.mask2[:,4,:,:]).unsqueeze(1))) + \
                self.layer4_3(laterals[2] * (self.mask3[:,4,:,:]).unsqueeze(1)) + \
                self.layer4_4(laterals[3] * (self.mask4[:,0,:,:]+self.mask4[:,4,:,:]).unsqueeze(1))
        layer4 = self.cbam_layer4(layer4)

        # layer 3     
        layer3 = self.layer3_1_1(self.layer3_1_2(laterals[0] * (self.mask1[:,3,:,:]).unsqueeze(1))) + \
                    self.layer3_2(laterals[1] * ((self.mask2[:,3,:,:]).unsqueeze(1))) + \
                    self.layer3_3(laterals[2] * (self.mask3[:,0,:,:]+self.mask3[:,3,:,:]).unsqueeze(1)) + \
                    self.layer3_4(F.interpolate(laterals[3] * (self.mask4[:,3,:,:]).unsqueeze(1), scale_factor=2))
        layer3 = self.cbam_layer3(layer3)

        # layer 2
        layer2 = self.layer2_1(laterals[0] * (self.mask1[:,2,:,:]).unsqueeze(1)) + \
                 self.layer2_2(laterals[1] * (self.mask2[:,0,:,:]+self.mask2[:,2,:,:]).unsqueeze(1)) + \
                 self.layer2_3(F.interpolate(laterals[2] * (self.mask3[:,2,:,:]).unsqueeze(1), scale_factor=2)) + \
                 self.layer2_4(F.interpolate(laterals[3] * (self.mask4[:,2,:,:]).unsqueeze(1), scale_factor=4))
        layer2 = self.cbam_layer2(layer2)
        
        # layer 1
        layer1 = self.layer1_1(laterals[0] * (self.mask1[:,0,:,:]+self.mask1[:,1,:,:]).unsqueeze(1)) + \
                self.layer1_2(F.interpolate(laterals[1] * (self.mask2[:,1,:,:]).unsqueeze(1), scale_factor=2)) + \
                self.layer1_3(F.interpolate(laterals[2] * (self.mask3[:,1,:,:]).unsqueeze(1), scale_factor=4)) + \
                self.layer1_4(F.interpolate(laterals[3] * (self.mask4[:,1,:,:]).unsqueeze(1), scale_factor=8))
        layer1 = self.cbam_layer1(layer1)
        
        layers = [layer1, layer2, layer3, layer4]

        
        # build upsample layers and outputs
        outs = [
            self.fpn_convs[i](layers[i]) for i in range(used_backbone_levels)
        ]

        out1 = self.upsample_layer_1(lowfeature,0, outs[0])
        

        lateral_layer1_1  = self.lateral_layer1_1(torch.cat((F.avg_pool2d(lowfeature, kernel_size=2, stride=2), inputs[0]), dim=1))
        out2 = self.upsample_layer_2(lateral_layer1_1, lowfeature, outs[1])
        lateral_layer2_1 = self.lateral_layer2_1(torch.cat((F.avg_pool2d(lowfeature, kernel_size=4, stride=4),
                                                            F.avg_pool2d(inputs[0], kernel_size=2, stride=2),
                                                            inputs[1]), dim=1))

        out3 = self.upsample_layer_3(lateral_layer2_1, lateral_layer1_1, outs[2])

        lateral_layer3 = self.lateral_layer3(torch.cat((F.avg_pool2d(inputs[0], kernel_size=4, stride=4),
                                                            F.avg_pool2d(inputs[1], kernel_size=2, stride=2),
                                                            inputs[2]), dim=1))
        
        out4 = self.upsample_layer_4(lateral_layer3, lateral_layer2_1, outs[3])


        outs = [out1,out2, out3, out4]

        if img_metas is not None:
            return tuple(outs), tuple(self.masks)
        else:
            return tuple(outs)

class LowLevel_FeaEnh(nn.Module):
    def __init__(self, out_channels, is_firstlevel=False):
        super(LowLevel_FeaEnh, self).__init__ ()
        self.outchannels = out_channels
        self.is_firstlevel = is_firstlevel
        self.deconv1 = ConvModule(out_channels, out_channels, 4,
                                          stride=2, padding=1, conv_cfg=dict(type='deconv'),
                                          norm_cfg=dict(type='BN'))
        self.deconv2 = ConvModule(out_channels, out_channels, 4,
                                          stride=2, padding=1, conv_cfg=dict(type='deconv'),
                                          norm_cfg=dict(type='BN'))
        self.conv1 = ConvModule(out_channels, out_channels, 3, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        
        self.conv3 = ConvModule(out_channels, out_channels, 3, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        if not is_firstlevel:
            self.conv2 = ConvModule(out_channels, out_channels, 3, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
            self.conv4 = ConvModule(out_channels, out_channels, 3, padding=1, 
                                          conv_cfg=dict(type='Conv2d'),norm_cfg=dict(type='BN'))
        
    def forward(self, low1, low2, high):
        high = self.deconv1(high)
        high = self.conv1(high)
        high = torch.sigmoid(torch.mean(high, dim=1).unsqueeze(1)) * low1 + high
        high = self.conv3(high)
        if not self.is_firstlevel:
            high = self.conv2(self.deconv2(high))
            high = torch.sigmoid(torch.mean(high, dim=1).unsqueeze(1)) * low2 + high
            high = self.conv4(high)
        else:
            high = self.deconv2(high)
        return high





