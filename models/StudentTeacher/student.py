import timm
import torch.nn as nn
import torch.nn.functional as F

class studentTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_indices=[2]
    ):
        super(studentTimm, self).__init__()     
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        
    def forward(self, x):
        
        features_t = self.feature_extractor(x)
    
        return features_t