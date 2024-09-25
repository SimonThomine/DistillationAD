import os
import torch
from models.trainer_base import BaseTrainer
from models.EfficientAD.efficientAD import loadPdnTeacher
from models.EfficientAD.efficientAD import get_pdn_small,get_pdn_medium
from utils.functions import cal_loss_quantile
from utils.util import loadWeights

class EadTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        assert self.modelName in ["small","medium"], "Model name must be either 'small' or 'medium'"
    
    def load_optim(self):
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 

    def load_model(self):
        loadPdnTeacher(self)
        if self.modelName=="small":
            self.student = get_pdn_small().to(self.device) 
        if self.modelName=="medium" : 
            self.student = get_pdn_medium().to(self.device)
            

    def change_mode(self, period="train"):
      if period == "train":
        self.student.train()
      else:
        self.student.eval()

    def infer(self,image):
        self.features_t = [self.teacher(image)]
        self.features_s=[self.student(image)]

    def computeLoss(self):
        loss=cal_loss_quantile(self.features_s, self.features_t,self.norm)
        return loss

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        
    def load_weights(self):
        self.student=loadWeights(self.student,self.model_dir,"student.pth")