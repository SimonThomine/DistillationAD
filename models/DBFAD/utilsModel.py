import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)

class conv3BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, stride, padding):
        super(conv3BnRelu, self).__init__()
        self.conv = conv3x3(in_chan, out_chan, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x



class conv1BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, stride, padding):
        super(conv1BnRelu, self).__init__()
        self.conv = conv1x1(in_chan, out_chan, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.softmax(x3)

        return x3 * x1


class Attention2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.softmax(x2)

        return x2 * x1




class BasicBlockDe(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockDe, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1_output = None
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        self.conv1_output = out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out