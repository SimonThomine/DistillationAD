import os
import torch
from models.trainer_base import BaseTrainer
from models.RememberingNormality.RD.teacherRD import resnetTeacherRD
from models.RememberingNormality.RD.de_resnetRM import de_resnetMemory
from utils.functions import cal_loss_cosine,cal_loss,cal_loss_orth
from utils.util import loadWeights

class RnrdTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        assert self.lambda1 is not None and self.lambda2 is not None and self.embedDim is not None, \
            "lambda1,lambda2 and embedDim must be defined in TrainingData"
        
    def load_optim(self):
        self.optimizer = torch.optim.Adam(list(self.student.parameters())+list(self.bn.parameters()), lr=self.lr, betas=(0.5, 0.999)) 

    def load_model(self):
        self.teacher,self.bn=resnetTeacherRD(backbone_name=self.modelName)
        self.teacher=self.teacher.to(self.device)
        self.bn=self.bn.to(self.device)
        self.student=de_resnetMemory(backbone_name=self.modelName,embedDim=self.embedDim).to(self.device)
    
    def load_iter(self):
        self.data_iter = iter(self.train_loader)
      
      
      
    def change_mode(self, period="train"):
      if period == "train":
        self.student.train()
        self.bn.train()
      else:
        self.student.eval()  
        self.bn.eval()
      
    def prepare(self):
        try:
            sample_next = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            sample_next = next(self.data_iter)
        self.imageExamplar = sample_next["imageBase"].to(self.device)
        

    def infer(self,image,test=False):
        if not test:
            features_t_examplar = self.teacher(self.imageExamplar)
            features_t_examplar = [features_t_examplar[1],features_t_examplar[2],features_t_examplar[3]]
            
            features_t_examplar_norm=[self.student.memory2(features_t_examplar[0]),
                                    self.student.memory1(features_t_examplar[1]),
                                    self.student.memory0(features_t_examplar[2])]
            
            features_t = self.teacher(image)
            features_t= [features_t[0],features_t[1],features_t[2]]
            embed=self.bn(features_t)
            features_s=self.student(embed)
            
            self.features_t_examplar=features_t_examplar
            self.features_t_examplar_norm=features_t_examplar_norm
            
        else : 
            features_t = self.teacher(image)
            features_t= [features_t[0],features_t[1],features_t[2]]
            embed=self.bn(features_t)
            features_s=self.student(embed)
        self.features_s=features_s
        self.features_t=features_t
        
        

    def computeLoss(self):
        loss_KD=cal_loss_cosine(self.features_s, self.features_t,self.norm)
        loss_NM=cal_loss(self.features_t_examplar, self.features_t_examplar_norm,self.norm)
        loss_ORTH=cal_loss_orth(self.student,rd=True)
        loss=loss_KD+self.lambda1*loss_NM +self.lambda2*loss_ORTH
        return loss

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        
        state = {"model": self.bn.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "bn.pth"))
        
    def load_weights(self):
        self.student=loadWeights(self.student,self.model_dir,"student.pth")
        self.bn=loadWeights(self.bn,self.model_dir,"bn.pth")