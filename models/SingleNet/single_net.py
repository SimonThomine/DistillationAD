import torch.nn as nn
from models.SingleNet.ffc import FFC_BN_ACT
from models.SingleNet.ffcT import FFC_T_BN_ACT


class singleNet(nn.Module):
    def __init__(self,inputsFilters=128):
        super().__init__() 
        
        self.reduceFactor=4
        self.context=nn.Sequential(
                    FFC_BN_ACT(inputsFilters, inputsFilters, kernel_size=3, ratio_gin=0, ratio_gout=0.5, stride=1, 
                               padding=1, bias=False,activation_layer=nn.LeakyReLU,momentumbs=0.01),
                    FFC_T_BN_ACT(inputsFilters, inputsFilters, kernel_size=4, ratio_gin=0.5, ratio_gout=0.5, stride=2,
                                 padding=1, bias=False,activation_layer=nn.LeakyReLU,momentumbs=0.01),
                    FFC_T_BN_ACT(inputsFilters, inputsFilters, kernel_size=4, ratio_gin=0.5, ratio_gout=0, stride=2,
                                 padding=1, bias=False,activation_layer=nn.LeakyReLU,momentumbs=0.01)
        )

    def forward(self, features): 
        subSampl=self.inverseTranspose(features[0])
        context=self.context(subSampl)[0] 
        return [context]

    def inverseTranspose(self,feat):
        featReduced=feat.clone()
        loop=self.reduceFactor//2
        for i in range(loop):
            featReduced=featReduced[:,:,1::2,1::2]
        return featReduced
    
    