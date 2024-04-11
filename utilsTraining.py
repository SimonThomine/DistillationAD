import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from models.teacher import teacherTimm
from models.StudentTeacher.student  import studentTimm
from datasets.mvtec import MVTecDataset
from models.EfficientAD.efficientAD import loadPdnTeacher
from models.EfficientAD.common import get_pdn_medium,get_pdn_small
from models.ReverseDistillation.rd import loadBottleNeckRD, loadStudentRD
from models.DBFAD.reverseResidual import reverse_student18

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
    else:
        raise Exception("Invalid distillation type :  Choices are ['st', 'ead','rd', 'dbfad']")
    
    # load bottleneck rd
        
        
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
    
    
def infer(trainer, img):
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
    return features_s,features_t


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
