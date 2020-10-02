import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift
import numpy as np
import random
import torchvision

#debug = True
debug = False

class VGG_Feat(nn.Module):
    """Using first 4/6/8 layers of VGG Netowrk for the RGB part"""
    def __init__(self, depth):
        super(VGG_Feat, self).__init__()
        self.depth = depth
        self.vgg_net = torchvision.models.vgg16(pretrained=True).features[:self.depth]
    def forward(self, rgb):
        rgb_feat = self.vgg_net(rgb)
        return rgb_feat

class Self_Attn(nn.Module):
    """Implemeted using conv layers"""
    def __init__(self, in_channels):
        #super(Self_Attn, self).__init__()
        #self.conv_attn1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size = 3, padding=1), nn.PReLU(num_parameters=1, init=0.2))
        #self.conv_attn2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, padding=1), nn.PReLU(init=0.9), nn.InstanceNorm2d(64))
        #self.conv_attn3 = nn.Sequential(nn.Conv2d(64, in_channels, kernel_size = 3, padding=1), nn.BatchNorm2d(in_channels), nn.PReLU(num_parameters=1, init=0.2))
        
        super(Self_Attn, self).__init__()
        self.conv_attn1 = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size = 3, padding=1), nn.PReLU(num_parameters=1, init=0.2))
        self.conv_attn2 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = 3, padding=1), nn.PReLU(init=0.9), nn.InstanceNorm2d(512))
        self.conv_attn3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size = 3, padding=1), nn.PReLU(init=0.9), nn.InstanceNorm2d(256))
        self.conv_attn4 = nn.Sequential(nn.Conv2d(256, in_channels, kernel_size = 3, padding=1), nn.InstanceNorm2d(in_channels), nn.PReLU(init=0.9))
    def forward(self, features):
        x = self.conv_attn1(features)
     #   np.save('x_1', x.detach().float().cpu().numpy().sum(1)[0])
        x = self.conv_attn2(x)
   #     np.save('x_2', x.detach().float().cpu().numpy().sum(1)[0])
        x = self.conv_attn3(x)
        # np.save('x_3', x.detach().float().cpu().numpy().sum(1)[0])
        x = self.conv_attn4(x)
        return x

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            print('selecting scale fact 8')
            stride = 1
            padding = 1
            kernel_size = 3

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()


        for idx in range(self.num_groups):
            self.upBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.attn = Self_Attn(128)     
        
    def forward(self, x, rgbf):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False
        
        # print('x ', x.shape)

        # print('self last hidden', self.last_hidden.shape)
        x = torch.cat((x, self.last_hidden), dim=1)

        x = self.compress_in(x)
        # print('compressed x', x.shape)
        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                 LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)
        if debug:
             print('gamma', self.gamma)
        attn = self.attn(torch.cat([output, rgbf], 1)) 
#the scale factor of 3 has been used accidently before trainnig and hence carried forward
#yet it doesnot have any effet as it gets compensated by self.gamma value
        fused = output * attn[:,:64,:,:] + rgbf * self.gamma * attn[:,64:,:,:] 
        self.last_hidden = fused
        return fused, attn

    def reset_state(self):
        self.should_reset = True

class SRFBN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type = 'prelu', norm_type = None):
        super(SRFBN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6

        elif upscale_factor == 4:
            stride = 1
            padding = 2
            kernel_size = 3
        elif upscale_factor == 8:
            stride = 1
            padding = 3
            kernel_size = 3

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor



        # LR feature extraction block
        self.conv_in1 = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in1 = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        #  conv2 is for rgb
        self.conv_in2 = ConvBlock(3, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_in2_down = ConvBlock(4*num_features,num_features,
                                 kernel_size=3,stride=2,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in2 = ConvBlock(num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)


        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type)
        # reconstruction block
		# uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = ConvBlock(num_features, num_features,
                               kernel_size=3, stride=stride,
                               act_type='prelu', norm_type=norm_type)
        
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type='prelu', norm_type=norm_type)

        self.attn = Self_Attn(num_features * 2)
        self.vgg_feat = VGG_Feat(4)
        self.maxpool = nn.MaxPool2d(2)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.add_mean = MeanShift(rgb_mean, rgb_std, 1)
        # self.interpolate_conv = ConvBlock(1, 1,
        #                          kernel_size=3,
        #                          act_type=act_type, norm_type=norm_type)

    def forward(self, x, rgb):
        self._reset_state()

        # we are not using below operation because our LR and HR sizes are same
        # inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        # print('before interpolate shape', x.shape)
        # inter_res = self.interpolate_conv(x)

        ILR = x

        demf = self.conv_in1(x)
        # print('after shape conv_in', x.shape)

        # print('Conv in shape', x.shape)
        demf = self.feat_in1(demf)
        
        if debug:
                for i in range(64):
                    np.save('before_demf_channel_{}'.format(i), demf.detach().float().cpu().numpy()[0][i])
        rgbf = self.vgg_feat(rgb)
        rgbf = self.maxpool(rgbf)
        ############################## LR block over ##################################
        
        if debug:
            for i in range(10):
                np.save('rgb_channel_{}'.format(i), rgbf.detach().float().cpu().numpy()[0][i])
        outputs = []
        
        for step_no in range(self.num_steps):
            if debug:
                for i in range(64):
                    np.save('demf_step_{}_channel_{}'.format(step_no, i), demf.detach().float().cpu().numpy()[0][i])
             
            fused, attn = self.block(demf, rgbf)
            #print('attn shape', attn.shape)
            #attn = self.attn(h)
            max_demf_attn, _ = torch.max(attn[:,:64,:,:], 1)
            max_rgbf_attn, _ = torch.max(attn[:,64:,:,:],1)

            if debug:
                print('channel pool dim demf', max_demf_attn.shape)
                np.save('max_demf_attn_step_no_{}'.format(step_no), max_demf_attn.detach().cpu().numpy()[0]) 
                np.save('max_rgbf_attn_step_no_{}'.format(step_no), max_rgbf_attn.detach().cpu().numpy()[0])
            if debug: 
                for i in range(10):
                    np.save('attn_map_step_no_{}_channel_{}'.format(step_no, i), attn.detach().float().cpu().numpy()[0][i])
                for i in range(64,74):
                    np.save('attn_map_step_no_{}_channel_{}'.format(step_no, i), attn.detach().float().cpu().numpy()[0][i])

            #fused = demf * attn[:,:64,:,:] + rgbf * self.gamma * attn[:,64:,:,:] 
            if debug: 
                for i in range(10):
                      np.save('fused_step_{}_channel_{}'.format(step_no, i), fused.detach().float().cpu().numpy()[0][i])
            
            residual = self.conv_out(self.out(fused))
            out = torch.add(ILR, residual)

            outputs.append(out)
        return outputs # return output of every timesteps
    
    def _reset_state(self):
        self.block.reset_state()

