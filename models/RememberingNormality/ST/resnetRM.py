import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from models.RememberingNormality.memoryModule import memoryModule
from models.RememberingNormality.utilsResnet import conv1x1, BasicBlock, Bottleneck,init_weights


resnet_layers = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3],
    "wide_resnet50_2": [3, 4, 6, 3],
    "wide_resnet101_2": [3, 4, 23, 3]
    }

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        embedDim: int,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],firstLayer=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

                
        if (block==BasicBlock):
            self.memory1 = memoryModule(L=embedDim,channel=64)
            self.memory2 = memoryModule(L=embedDim,channel=128)
        else :
            self.memory1 = memoryModule(L=embedDim,channel=256)
            self.memory2 = memoryModule(L=embedDim,channel=512)
                
        init_weights(self)  

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,firstLayer: bool = False) -> nn.Sequential:
        if (not firstLayer):
            inplanes=self.inplanes*2 
        else :
            inplanes=self.inplanes
            
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride), 
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, self.groups, 
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        feature_a = self.layer1(x) 
        feature_mem_a = self.memory1(feature_a)
        feature_a_cat=torch.cat((feature_a,feature_mem_a),1)
        
        feature_b = self.layer2(feature_a_cat)
        
        feature_mem_b = self.memory2(feature_b)
        feature_b_cat=torch.cat((feature_b,feature_mem_b),1)
        
        feature_c = self.layer3(feature_b_cat)
        
        return [feature_a, feature_b, feature_c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(block: Type[Union[BasicBlock, Bottleneck]],layers: List[int],embedDim: int,**kwargs: Any) -> ResNet:
    model = ResNet(block, layers,embedDim, **kwargs)
    return model

def resnetMemory(backbone_name,embedDim=50,**kwargs):
    if backbone_name=="resnet18" or backbone_name=="resnet34":
        return _resnet(BasicBlock, resnet_layers[backbone_name],embedDim)
    elif backbone_name=="resnet50" or backbone_name=="resnet101" or backbone_name=="resnet152":
        return _resnet(Bottleneck, resnet_layers[backbone_name],embedDim)
    elif backbone_name=="wide_resnet50_2" or backbone_name=="wide_resnet101_2":
        kwargs['width_per_group'] = 64 * 2
        return _resnet(Bottleneck, resnet_layers[backbone_name],embedDim,**kwargs)
    else:
        raise Exception("Invalid model name :  Choices are ['resnet18', 'resnet34','resnet50', 'resnet101',\
                        'resnet152', 'wide_resnet50_2', 'wide_resnet101_2']")