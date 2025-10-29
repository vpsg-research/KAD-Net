from . import *
from network.kanarchs import KANBlock, PatchEmbed
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import Module, Parameter, Softmax
from .Random_Noise import Random_Noise

class ST_Encoder(nn.Module):
    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(ST_Encoder, self).__init__()

        
        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        # KAN-based Feature Extraction
        self.patch_embed = PatchEmbed(img_size=32, patch_size=3, stride=1, in_chans=256, embed_dim=256)
        self.kan_block = KANBlock(dim=256)

        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message0 = ConvBlock(1, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)

        self.message_length = message_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def forward(self, x, watermark):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # KAN-based Feature Extraction
        d4, H, W = self.patch_embed(d4)  
        d4 = self.kan_block(d4, H, W)  
        d4 = d4.transpose(1, 2).view(d4.shape[0], 256, H, W)  


        # Watermark Embedding
        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]), mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        u3 = torch.cat((d3, u3, expanded_message), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]), mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)
        u2 = torch.cat((d2, u2, expanded_message), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]), mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        u1 = torch.cat((d1, u1, expanded_message), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        expanded_message = self.linear0(watermark)
        expanded_message = expanded_message.view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]), mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        u0 = torch.cat((d0, u0, expanded_message), dim=1)
        u0 = self.att0(u0)

        
        image = self.Conv_1x1(torch.cat((x, u0), dim=1))

        forward_image = image.clone().detach()
        
        gap = forward_image.clamp(-1, 1) - forward_image

        return image + gap


class ST_Decoder(nn.Module):

    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(ST_Decoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)
        
        # KAN-based Feature Extraction
        self.patch_embed = PatchEmbed(img_size=32, patch_size=3, stride=1, in_chans=256, embed_dim=256)
        self.kan_block = KANBlock(dim=256)

        self.up3 = UP(256, 128)
        self.att3 = ResBlock(128 * 2, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.att2 = ResBlock(64 * 2, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.att1 = ResBlock(32 * 2, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.att0 = ResBlock(16 * 2, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.message_layer = nn.Linear(message_length * message_length, message_length)
        self.message_length = message_length


    def forward(self, x):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        d4 = self.down4(d3)
        
        d4, H, W = self.patch_embed(d4)  
        d4 = self.kan_block(d4, H, W)  
        d4 = d4.transpose(1, 2).view(d4.shape[0], 256, H, W)  

        u3 = self.up3(d4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = self.att3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = self.att2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = self.att1(u1)

        u0 = self.up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = self.att0(u0)

        residual = self.Conv_1x1(u0)
        message = F.interpolate(residual, size=(self.message_length, self.message_length),
                                                           mode='nearest')
        message = message.view(message.shape[0], -1)
        message = self.message_layer(message)

        return message



#Structure-aware Difference Module
"""Structure-aware Difference Module (SDM) integrates multi-directional
differential convolutions into equivalent kernels to enhance edge forgery perception 
and improve watermark recovery under complex distortions."""
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self, device):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3, device=device).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self, device):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x, device):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5, device=device).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self, device):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3, device=device).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self, device):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3, device=device).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class SDMConv(nn.Module):
    def __init__(self, dim):
        super(SDMConv, self).__init__()
        self.conv1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)  

        self.bn = nn.BatchNorm2d(dim)  

    def forward(self, x):
        device = x.device
        w1, b1 = self.conv1.get_weight(device)
        w2, b2 = self.conv2.get_weight(device)
        w3, b3 = self.conv3.get_weight(device)
        w4, b4 = self.conv4.get_weight(device)
        w5, b5 = self.conv5.weight, self.conv5.bias 

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5

        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        res = self.bn(res)  
        return res


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__()
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)


class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.deconv = SDMConv(out_channels)  

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  
        x = self.conv(x)  
        x = self.deconv(x)  
        return x
    
    
# Source Tracing Encoder–Noise Layer–Decoder Branch
class ST_EncoderDecoder(nn.Module):

	def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder):
		super(ST_EncoderDecoder, self).__init__()
		self.encoder = ST_Encoder(message_length, attention = attention_encoder)
		self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
		self.decoder_C = ST_Decoder(message_length, attention = attention_decoder)
		

	def forward(self, image, message, mask):
		encoded_image = self.encoder(image, message)
		noised_image_C, noised_image_R, noised_image_F = self.noise([encoded_image, image, mask])
		decoded_message_C = self.decoder_C(noised_image_C)
		return encoded_image, noised_image_C, decoded_message_C