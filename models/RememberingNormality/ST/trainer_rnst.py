import os
import torch
from models.trainer_base import BaseTrainer
from models.RememberingNormality.ST.teacherST import resnetTeacherST
from models.RememberingNormality.ST.resnetRM import resnetMemory
from utils.functions import cal_loss_cosine,cal_loss,cal_loss_orth
from utils.util import loadWeights

class RnstTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        assert self.lambda1 is not None and self.lambda2 is not None and self.embedDim is not None, \
            "lambda1,lambda2 and embedDim must be defined in TrainingData"
        
        
    def load_optim(self):
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 

    def load_model(self):
        self.teacher=resnetTeacherST(backbone_name=self.modelName).to(self.device)
        self.student=resnetMemory(backbone_name=self.modelName,embedDim=self.embedDim).to(self.device)
        
        
    def load_iter(self):
        self.data_iter = iter(self.train_loader)
      
      
    def change_mode(self, period="train"):
      if period == "train":
        self.student.train()
      else:
        self.student.eval()  
      
    def prepare(self):
        try:
            sample_next = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            sample_next = next(self.data_iter)
        self.imageExamplar = sample_next["imageBase"].to(self.device)
        

    def infer(self,image,test=False):
        if (not test):
            features_t_examplar = self.teacher.forward_normality_embedding(self.imageExamplar)
            features_t_examplar = [features_t_examplar[1],features_t_examplar[2]]
            features_t_examplar_norm=[self.student.memory1(features_t_examplar[0]),
                                    self.student.memory2(features_t_examplar[1])]
            features_t = self.teacher(image)
            features_s=self.student(image)
            self.features_t_examplar=features_t_examplar
            self.features_t_examplar_norm=features_t_examplar_norm
        else : 
            features_t = self.teacher(image)
            features_s=self.student(image)
        self.features_s=features_s
        self.features_t=features_t
        

    def computeLoss(self):
        loss_KD=cal_loss_cosine(self.features_s, self.features_t,self.norm)
        loss_NM=cal_loss(self.features_t_examplar, self.features_t_examplar_norm,self.norm)
        loss_ORTH=cal_loss_orth(self.student,rd=False)
        loss=loss_KD+self.lambda1*loss_NM +self.lambda2*loss_ORTH
        return loss

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        
    def load_weights(self):
        self.student=loadWeights(self.student,self.model_dir,"student.pth")