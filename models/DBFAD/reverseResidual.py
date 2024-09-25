import torch.nn as nn
from models.DBFAD.utilsModel import  conv3BnRelu, conv1BnRelu, Attention, Attention2, BasicBlockDe,deconv2x2
from models.sspcab import SSPCAB
from models.utilsResnet import init_weights


class ReverseStudent(nn.Module):
    def __init__(self, block, layers,groups=1,DG=False):
        super(ReverseStudent, self).__init__()

        self.DG=DG
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 512  
        
        self.dilation = 1
            
        self.groups = groups
        self.base_width = 64

        self.block1 = nn.Sequential(
            
            conv1BnRelu(64, 128, stride=2, padding=0), 
            conv1BnRelu(128, 256, stride=2, padding=0)
        )
        self.block2 = nn.Sequential(
            conv1BnRelu(64, 128, stride=2, padding=0),
            conv1BnRelu(128, 256, stride=2, padding=0),
        )
        self.block3 = nn.Sequential(
            conv1BnRelu(128, 256, stride=2, padding=0)
        )
        self.block4 = nn.Sequential(
            conv1BnRelu(256, 256, stride=1, padding=0)
        )
        self.block5 = nn.Sequential(
            conv1BnRelu(256, 256, stride=1, padding=0),
            conv1BnRelu(256, 256, stride=1, padding=0),
            conv1BnRelu(256, 512, stride=2, padding=0),
            SSPCAB(512)
        )
        self.AttBlock = nn.Sequential(
            Attention2(512, 512)
        )
        if not self.DG:
            self.residualLayer0_1 = nn.Sequential(
                conv3BnRelu(64, 64, stride=1, padding=1),
                Attention(64, 64)
            )
            self.residualLayer1_2 = nn.Sequential(
                conv3BnRelu(64, 128, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=2, stride=2),
                Attention(128, 128)
            )
            self.residualLayer2_3 = nn.Sequential(
                conv3BnRelu(128, 256, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=2, stride=2),
                Attention(256, 256)
            )


        self.layer0 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2])

        init_weights(self)


    def _make_layer(self, block: BasicBlockDe, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out1 = self.block1(x[0])
        out2 = self.block2(x[1])
        out3 = self.block3(x[2])
        out4 = self.block4(x[3])
        out = (out1 + out2 + out3 + out4)
        out = self.block5(out)
        out = self.AttBlock(out)

        if not self.DG:
            resi1 = self.residualLayer0_1(x[0])
            resi2 = self.residualLayer1_2(x[1])
            resi3 = self.residualLayer2_3(x[2])

        feature_x = self.layer0(out)
        feature_x = feature_x + resi3 if not self.DG else feature_x
        feature_a = self.layer1(feature_x)  
        feature_a = feature_a+resi2 if not self.DG else feature_a
        feature_b = self.layer2(feature_a)  
        feature_b = feature_b+resi1 if not self.DG else feature_b
        feature_c = self.layer3(feature_b)
        return feature_c, feature_b, feature_a, feature_x

    def forward(self, x):
        x, x1, x2, x3 = self._forward_impl(x)
        return x, x1, x2, x3


def reverse_student(block, layers,DG, **kwargs):
    model = ReverseStudent(block, layers,DG=DG, **kwargs)
    return model


def reverse_student18(DG=False, **kwargs):
    return reverse_student(BasicBlockDe, [2, 2, 2, 2],DG=DG,
                           **kwargs)