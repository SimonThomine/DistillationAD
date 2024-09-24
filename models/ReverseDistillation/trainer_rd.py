import os
import torch
from models.trainer_base import BaseTrainer
from models.teacher import teacherTimm
from models.ReverseDistillation.rd import loadBottleNeckRD,loadStudentRD
from utils.functions import cal_loss
from utils.util import loadWeights

class RdTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        
        
        
    def load_optim(self):
        self.optimizer = torch.optim.Adam(list(self.student.parameters())+list(self.bn.parameters()), lr=self.lr, betas=(0.5, 0.999)) 


    def load_model(self):
        self.teacher=teacherTimm(backbone_name=self.modelName,out_indices=[1,2,3]).to(self.device)
        loadBottleNeckRD(self)
        loadStudentRD(self)
        
        
    def change_mode(self, period="train"):
      if period == "train":
        self.student.train()
        self.bn.train()
      else:
        self.student.eval()  
        self.bn.eval()    

    def infer(self,image):
        self.features_t = self.teacher(image)
        embed=self.bn(self.features_t)
        self.features_s=self.student(embed)

    def computeLoss(self):
        loss=cal_loss(self.features_s, self.features_t,self.norm)
        return loss

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        state = {"model": self.bn.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "bn.pth"))
        
    def load_weights(self):
        self.student=loadWeights(self.student,self.model_dir,"student.pth")
        self.bn=loadWeights(self.bn,self.model_dir,"bn.pth")