import os
import torch
import torch.nn.functional as F
from models.trainer_base import BaseTrainer
from models.DBFAD.reverseResidual import reverse_student18
from models.teacher import teacherTimm
from utils.functions import cal_loss
from utils.util import loadWeights

class DbfadTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        
        
    def load_optim(self):
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 


    def load_model(self):
        self.teacher=teacherTimm(backbone_name="resnet34",out_indices=[0,1,2,3]).to(self.device)
        self.student=reverse_student18().to(self.device)

    def change_mode(self, period="train"):
      if period == "train":
        self.student.train()
      else:
        self.student.eval()

    def infer(self,image):
        self.features_t = self.teacher(image)
        self.features_t = [F.max_pool2d(self.features_t[0],kernel_size=3,stride=2,padding=1),self.features_t[1],self.features_t[2],self.features_t[3]]
        self.features_s=self.student(self.features_t)

    def computeLoss(self):
        loss=cal_loss(self.features_s, self.features_t,self.norm)
        return loss

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        
    def load_weights(self):
        self.student=loadWeights(self.student,self.model_dir,"student.pth")
      
    