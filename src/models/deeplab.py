import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[3], dilation=rates[3])

    def forward(self, x):
        conv1x1 = self.conv1x1_1(x)
        conv3x3_1 = self.conv3x3_1(x)
        conv3x3_2 = self.conv3x3_2(x)
        conv3x3_3 = self.conv3x3_3(x)
        conv3x3_4 = self.conv3x3_4(x)

        out = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, conv3x3_4], dim=1)

        return out

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.resnet_backbone = torchvision.models.resnet18(pretrained=True)
        self.aspp = ASPP(in_channels=512, out_channels=256)  # Adjust in_channels based on the ResNet version
        self.conv1x1 = nn.Conv2d(1280, 256, kernel_size=1)  # Adjust the number of input channels here
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder
        x = self.resnet_backbone.conv1(x)
        x = self.resnet_backbone.bn1(x)
        x = self.resnet_backbone.relu(x)
        x = self.resnet_backbone.maxpool(x)

        x = self.resnet_backbone.layer1(x)
        x = self.resnet_backbone.layer2(x)
        x = self.resnet_backbone.layer3(x)
        x = self.resnet_backbone.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        # Decoder
        x = self.conv1x1(x)
        x = self.upsample(x)

        # Final segmentation head
        x = self.segmentation_head(x)

        return x