import os
import torch
from models.trainer_base import BaseTrainer
from models.teacher import teacherTimm
from models.StudentTeacher.student import studentTimm
from utils.functions import cal_loss
from utils.util import loadWeights

class MixedTrainer(BaseTrainer):
    def __init__(self, data, device):
        super().__init__(data, device)
        
    def load_optim(self):
        params=[]
        for student in self.students:
            params+=list(student.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999))
        
        
    def load_model(self):
        self.teachers=[]
        for model,indice in zip(self.modelName,self.outIndices):
            teacher=teacherTimm(backbone_name=model,out_indices=indice).to(self.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            self.teachers.append(teacher)
            
        self.students=[]
        for model,indice in zip(self.modelName,self.outIndices):
            student=studentTimm(backbone_name=model,out_indices=indice).to(self.device)
            self.students.append(student)   
        

    def change_mode(self, period="train"):
        if period == "train":
            for student in self.students:
                student.train()
        else:
            for student in self.students:
                student.eval()
    
    def infer(self,image):
        features_s=[]
        features_t=[]
        for teacher,student in zip(self.teachers,self.students):
            features_t.extend(teacher(image))
            features_s.extend(student(image))
        self.features_t = features_t
        self.features_s=features_s
        
        
    def computeLoss(self):
        loss=cal_loss(self.features_s, self.features_t,self.norm)
        return loss

    def save_checkpoint(self):
        for i,student in enumerate(self.students):
            state = {"model": student.state_dict()}
            torch.save(state, os.path.join(self.model_dir, "student"+str(i)+".pth"))
        
        
        
    def load_weights(self):
        for i,student in enumerate(self.students):
            self.student=loadWeights(student,self.model_dir,"student"+str(i)+".pth")