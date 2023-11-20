import torch
import torch.nn as nn
import torch.nn.functional as F

class AtrousSeparableConvolution(nn.Module):
    """Atrous Separable Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(AtrousSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)

    def forward(self, x):
        return self.pointwise(F.relu(self.depthwise(x)))

class DecoderBlock(nn.Module):
    """Decoder block"""

    def __init__(self, in_channels, out_channels, upsample_factor):
        super(DecoderBlock, self).__init__()
        self.atrous_separable_convolution_1 = AtrousSeparableConvolution(in_channels, in_channels // 2, 3, 1, dilation=2, padding=1)
        self.atrous_separable_convolution_2 = AtrousSeparableConvolution(in_channels // 2, out_channels, 3, 1, dilation=4, padding=1)
        self.upsample = nn.Upsample(scale_factor=upsample_factor)

    def forward(self, x):
        x = self.atrous_separable_convolution_1(x)
        x = self.atrous_separable_convolution_2(x)
        return self.upsample(x)

class DeepLabv3Plus(nn.Module):
    """DeepLabv3+"""

    def __init__(self, num_classes=21, backbone='resnet101'):
        super(DeepLabv3Plus, self).__init__()

        # Load the encoder backbone
        if backbone == 'resnet101':
            self.encoder = ResNet101()
        elif backbone == 'resnet50':
            self.encoder = ResNet50()
        else:
            raise NotImplementedError(f'Backbone "{backbone}" is not supported.')

        # Remove the last layer of the encoder
        del self.encoder.fc

        # Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = nn.ModuleDict([
            ('aspp_1', AtrousSeparableConvolution(2048, 256, 1, 1, dilation=1, padding=0)),
            ('aspp_2', AtrousSeparableConvolution(2048, 256, 3, 1, dilation=6, padding=1)),
            ('aspp_3', AtrousSeparableConvolution(2048, 256, 3, 1, dilation=12, padding=1)),
            ('aspp_4', AtrousSeparableConvolution(2048, 256, 3, 1, dilation=24, padding=1)),
        ])

        # Decoder
        self.decoder_block_1 = DecoderBlock(1280, 256, 2)
        self.decoder_block_2 = DecoderBlock(512, 128, 2)
        self.decoder_block_3 = DecoderBlock(256, 64, 2)
        self.decoder_block_4 = DecoderBlock(64, 64, 1)

        # Final classification layer
        self.classifier = nn.Conv2d(64, num_classes, 1, 1, padding=0)

    def forward(self, x):
        # Extract features from the encoder
        features = self.encoder(x)

        # ASPP
        aspp_outputs = {}
        for name, module in self.aspp.items():
            aspp_outputs[name] = module(features['layer4'])

        # Concatenate ASPP outputs and pass them through a convolution
        aspp_out = torch.cat([aspp_outputs[name] for name in self.aspp.keys()], dim=1)
        aspp_out = nn.Conv2d(1024, 256, 1