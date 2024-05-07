import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from utils.functions import (
    cal_loss,
    cal_loss_cosine,
    cal_loss_quantile,
    cal_loss_orth
)

from models.teacher import teacherTimm
from models.StudentTeacher.student  import studentTimm
from models.EfficientAD.efficientAD import loadPdnTeacher
from models.EfficientAD.common import get_pdn_medium,get_pdn_small
from models.ReverseDistillation.rd import loadBottleNeckRD, loadStudentRD
from models.DBFAD.reverseResidual import reverse_student18
from models.RememberingNormality.ST.resnetRM import resnetMemory
from models.RememberingNormality.ST.teacherST import resnetTeacherST
from models.RememberingNormality.RD.de_resnetRM import de_resnetMemory
from models.RememberingNormality.RD.teacherRD import resnetTeacherRD


def getParams(trainer,data,device):
    trainer.device = device
    trainer.validation_ratio = 0.2
    trainer.data_path = data['data_path']
    trainer.obj = data['obj']
    trainer.img_resize = data['TrainingData']['img_size']
    trainer.img_cropsize = data['TrainingData']['crop_size']
    trainer.num_epochs = data['TrainingData']['epochs']
    trainer.lr = data['TrainingData']['lr']
    trainer.batch_size = data['TrainingData']['batch_size']   
    trainer.save_path = data['save_path']
    trainer.model_dir = trainer.save_path+ "/models" + "/" + trainer.obj  
    trainer.img_dir = trainer.save_path+ "/imgs" + "/" + trainer.obj 
    trainer.modelName = data['backbone']
    trainer.outIndices = data['out_indice']
    trainer.distillType=data['distillType']
    trainer.norm = data['TrainingData']['norm']
    
    if trainer.distillType.startswith('rn'):
        trainer.embedDim = data['TrainingData']['embedDim']
        trainer.lambda1 = data['TrainingData']['lambda1']
        trainer.lambda2 = data['TrainingData']['lambda1']
        
        
def loadWeights(model,model_dir,alias):
    try:
        checkpoint = torch.load(os.path.join(model_dir, alias))
    except:
        raise Exception("Check saved model path.")
    model.load_state_dict(checkpoint["model"])
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    return model

def loadTeacher(trainer):
    if (trainer.distillType=="st"):
        trainer.teacher=teacherTimm(backbone_name=trainer.modelName,out_indices=trainer.outIndices).to(trainer.device)
    elif (trainer.distillType=="ead"):
        loadPdnTeacher(trainer)
    elif (trainer.distillType=="rd"):
        trainer.teacher=teacherTimm(backbone_name=trainer.modelName,out_indices=[1,2,3]).to(trainer.device)
        loadBottleNeckRD(trainer)
    elif (trainer.distillType=="dbfad"):
        trainer.teacher=teacherTimm(backbone_name="resnet34",out_indices=[0,1,2,3]).to(trainer.device)
    elif (trainer.distillType=="mixed"):
        trainer.teacher=teacherTimm(backbone_name=trainer.modelName[0],out_indices=trainer.outIndices[0]).to(trainer.device)
        trainer.teacher2=teacherTimm(backbone_name=trainer.modelName[1],out_indices=trainer.outIndices[1]).to(trainer.device)
        trainer.teacher2.eval()
        for param in trainer.teacher2.parameters():
            param.requires_grad = False
    elif (trainer.distillType=="rnst"):
        trainer.teacher=resnetTeacherST(backbone_name=trainer.modelName).to(trainer.device)
    elif (trainer.distillType=="rnrd"):
        trainer.teacher,trainer.bn=resnetTeacherRD(backbone_name=trainer.modelName)
        trainer.teacher=trainer.teacher.to(trainer.device)
        trainer.bn=trainer.bn.to(trainer.device)
    else:
        raise Exception("Invalid distillation type :  Choices are ['st', 'ead','rd', 'dbfad','rnst','rnrd]")
    
    trainer.teacher.eval()
    for param in trainer.teacher.parameters():
        param.requires_grad = False

def loadModels(trainer):
    if (trainer.distillType=="st"):
        loadTeacher(trainer)
        trainer.student=studentTimm(backbone_name=trainer.modelName,out_indices=trainer.outIndices).to(trainer.device)
    if (trainer.distillType=="ead"):
        loadTeacher(trainer)
        if trainer.modelName=="small":
            trainer.student = get_pdn_small().to(trainer.device) # 768 if autoencoder
        if trainer.modelName=="medium" : 
            trainer.student = get_pdn_medium().to(trainer.device) # 768 if autoencoder
    if (trainer.distillType=="rd"):
        loadTeacher(trainer)
        loadStudentRD(trainer)
    if (trainer.distillType=="dbfad"):
        loadTeacher(trainer)
        trainer.student=reverse_student18().to(trainer.device)
    if (trainer.distillType=="mixed"):
        loadTeacher(trainer)
        trainer.student=studentTimm(backbone_name=trainer.modelName[0],out_indices=trainer.outIndices[0]).to(trainer.device)
        trainer.student2=studentTimm(backbone_name=trainer.modelName[1],out_indices=trainer.outIndices[1]).to(trainer.device)
    if (trainer.distillType=="rnst"):
        loadTeacher(trainer)
        trainer.student=resnetMemory(backbone_name=trainer.modelName,embedDim=trainer.embedDim).to(trainer.device)
    if (trainer.distillType=="rnrd"):
        loadTeacher(trainer)
        trainer.student=de_resnetMemory(backbone_name=trainer.modelName,embedDim=trainer.embedDim).to(trainer.device)
    
def loadDataset(trainer):
    kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
    train_dataset = MVTecDataset(root_dir=trainer.data_path+"/"+trainer.obj+"/train/good",
        resize_shape=[trainer.img_resize,trainer.img_resize],
        crop_size=[trainer.img_cropsize,trainer.img_cropsize],
        phase='train'
    )
    img_nums = len(train_dataset)
    valid_num = int(img_nums * trainer.validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_num, valid_num]
    )
    trainer.train_loader=torch.utils.data.DataLoader(train_data, batch_size=trainer.batch_size, shuffle=True, **kwargs)
    trainer.val_loader=torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)
    
    
def infer(trainer, img,imgExamplar=None,test=False):
    if (trainer.distillType=="st" ):
        features_t = trainer.teacher(img)
        features_s=trainer.student(img)
    if (trainer.distillType=="ead"):
        features_t = [trainer.teacher(img)]
        features_s=[trainer.student(img)]
    if (trainer.distillType=="rd"):
        features_t = trainer.teacher(img)
        embed=trainer.bn(features_t)
        features_s=trainer.student(embed)
    if (trainer.distillType=="dbfad"):
        features_t = trainer.teacher(img)
        features_t = [F.max_pool2d(features_t[0],kernel_size=3,stride=2,padding=1),features_t[1],features_t[2],features_t[3]]
        features_s=trainer.student(features_t)
    if (trainer.distillType=="mixed"):
        features_t = trainer.teacher(img)
        features_t2 = trainer.teacher2(img)
        features_s=trainer.student(img)
        features_s2=trainer.student2(img) 
        features_s=list(features_s)+list(features_s2)
        features_t=list(features_t)+list(features_t2)
        
    if (trainer.distillType=="rnst"):  
        if (not test):
            features_t_examplar = trainer.teacher.forward_normality_embedding(imgExamplar)
            features_t_examplar = [features_t_examplar[1],features_t_examplar[2]]
            features_t_examplar_norm=[trainer.student.memory1(features_t_examplar[0]),
                                    trainer.student.memory2(features_t_examplar[1])]
            
            features_t = trainer.teacher(img)
            features_s=trainer.student(img)
            
            return features_s,features_t,features_t_examplar,features_t_examplar_norm
        else : 
            features_t = trainer.teacher(img)
            features_s=trainer.student(img)
    if (trainer.distillType=="rnrd"):  
        if (not test):
            features_t_examplar = trainer.teacher(imgExamplar)
            features_t_examplar = [features_t_examplar[1],features_t_examplar[2],features_t_examplar[3]]
            
            features_t_examplar_norm=[trainer.student.memory2(features_t_examplar[0]),
                                    trainer.student.memory1(features_t_examplar[1]),
                                    trainer.student.memory0(features_t_examplar[2])]

            
            features_t = trainer.teacher(img)
            features_t= [features_t[0],features_t[1],features_t[2]]
            embed=trainer.bn(features_t)
            features_s=trainer.student(embed)
            
            return features_s,features_t,features_t_examplar,features_t_examplar_norm
        else : 
            features_t = trainer.teacher(img)
            features_t= [features_t[0],features_t[1],features_t[2]]
            embed=trainer.bn(features_t)
            features_s=trainer.student(embed)
    return features_s,features_t


def computeLoss(trainer,features_s, features_t,features_t_examplar=None, features_t_examplar_norm=None):
    if (trainer.distillType.startswith("rn")):
        loss_KD=cal_loss_cosine(features_s, features_t,trainer.norm)
        loss_NM=cal_loss(features_t_examplar, features_t_examplar_norm,trainer.norm)
        if (trainer.distillType=="rnst"):
            loss_ORTH=cal_loss_orth(trainer.student,rd=False)
        else :
            loss_ORTH=cal_loss_orth(trainer.student,rd=True)
        loss=loss_KD+trainer.lambda1*loss_NM +trainer.lambda2*loss_ORTH
    elif(trainer.distillType=="ead"):
        loss=cal_loss_quantile(features_s, features_t,trainer.norm)
    else:
        loss=cal_loss(features_s, features_t,trainer.norm)
    return loss

def computeAUROC(scores,gt_list,obj,name="base"):
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(obj + " image"+str(name)+" ROCAUC: %.3f" % (img_roc_auc))
    return img_roc_auc,img_scores  

def cal_importance(ft, fs,norm):

    fs_norm = F.normalize(fs, p=2) if norm else fs
    ft_norm = F.normalize(ft, p=2) if norm else ft

    f_loss = 0.5 * (ft_norm - fs_norm) ** 2

    sumOverAxes=torch.sum(f_loss,dim=[1,2])
    sortedIndex=torch.argsort(sumOverAxes,descending=True)
    ft_norm = ft_norm[sortedIndex]
    fs_norm = fs_norm[sortedIndex]
    
    return ft_norm,fs_norm
