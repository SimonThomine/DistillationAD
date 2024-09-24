import torch
import torch.nn as nn


class deconvModified(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,stride,padding):
        super().__init__() 
        self.upSample=nn.UpsamplingNearest2d(scale_factor=stride)
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=1,padding=padding)
        
    def forward(self,x):
        upSample=self.upSample(x)
        output=self.conv(upSample)
        return output

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1,momentumbs=0.1):
        super().__init__()
        self.groups = groups
        
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        
        self.bn = torch.nn.BatchNorm2d(out_channels * 2,momentum=momentumbs)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, _, _, _ = x.size()

        fft_dim = (-2, -1)
        

        ffted = torch.fft.rfftn(x, dim=fft_dim)

        ffted = torch.stack((ffted.real, ffted.imag), dim=-1) 
        
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  

        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  

        ffted = torch.complex(ffted[..., 0], ffted[..., 1])   
        ifft_shape_slice = x.shape[-2:]
        
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim) 
        
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1,momentumbs=0.1):
        super().__init__()


        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2,momentum=momentumbs),
            nn.ReLU(inplace=True)
        )

        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups,momentumbs=momentumbs)
        self.conv2 = nn.ConvTranspose2d(out_channels // 2, out_channels,kernel_size=2, stride=stride, padding=0, groups=groups, bias=False) 
                
        
    def forward(self, x):
        
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(output) 

        return output


class FFC_T(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,momentumbs=0.1):
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg


        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.ConvTranspose2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.ConvTranspose2d 
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.ConvTranspose2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2,momentumbs=momentumbs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_T_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 momentumbs=0.1):
        super().__init__()
        self.ffc = FFC_T(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias,momentumbs=momentumbs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)),momentum=momentumbs)
        self.bn_g = gnorm(int(out_channels * ratio_gout),momentum=momentumbs)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g
