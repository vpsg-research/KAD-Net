from . import *
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.nn import Module, Parameter, Softmax
from .Random_Noise import Random_Noise

class FD_Encoder(nn.Module):
    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(FD_Encoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        self.danet_head = SAEM(256) 

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

        d4 = self.danet_head(d4)  

        # **Watermark Information Embedding**
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

        # **Generate the final image with embedded watermark**
        image = self.Conv_1x1(torch.cat((x, u0), dim=1))

        forward_image = image.clone().detach()

        gap = forward_image.clamp(-1, 1) - forward_image

        return image + gap

class FD_Decoder(nn.Module):
    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(FD_Decoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)

        self.down4 = Down(128, 256, blocks=blocks)
        
        self.danet_head = SAEM(256)

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
        
        d4 = self.danet_head(d4)  

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

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class Position_Feature_Extraction(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.op = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.op(x)

class Channel_Feature_Extraction(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.op = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv2d, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
class SAEM(nn.Module):
    def __init__(self, in_channels):
        super(SAEM, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                     nn.ReLU())

        self.conv5c = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1),
                                     nn.ReLU())

        self.sa = Position_Feature_Extraction(inter_channels)
        self.sc = Channel_Feature_Extraction(inter_channels)
        self.conv51 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                     nn.ReLU())
        self.conv52 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1),
                                     nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1),
                                   nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output1 = sa_conv + feat1

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output1 = sc_conv + feat2

        out = sa_output1 + sc_output1
        out = self.conv6(out)
        return out


# Forgery Detection Encoder–Noise Layer–Decoder Branch
class FD_EncoderDecoder(nn.Module):

	def __init__(self, message_length, noise_layers_R, noise_layers_F, attention_encoder, attention_decoder):
		super(FD_EncoderDecoder, self).__init__()
		self.encoder = FD_Encoder(message_length, attention = attention_encoder)
		self.noise = Random_Noise(noise_layers_R + noise_layers_F, len(noise_layers_R), len(noise_layers_F))
		self.decoder_RF = FD_Decoder(message_length, attention=attention_decoder)  

	def forward(self, image, message, mask):
		encoded_image = self.encoder(image, message)
		noised_image_C, noised_image_R, noised_image_F = self.noise([encoded_image, image, mask])
		decoded_message_R = self.decoder_RF(noised_image_R)
		decoded_message_F = self.decoder_RF(noised_image_F)
		return encoded_image, noised_image_C, decoded_message_R, decoded_message_F