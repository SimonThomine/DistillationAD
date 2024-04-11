import os
import time
import numpy as np
import torch
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
from utils.util import  AverageMeter,readYamlConfig,set_seed
from utils.functions import (
    cal_loss,
    cal_anomaly_maps,
)
from utilsTraining import getParams,loadWeights,loadModels,loadDataset,infer,computeAUROC

class NetTrainer:          
    def __init__(self, data,device):  
        
        getParams(self,data,device)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # You can set seed for reproducibility
        set_seed(42)
        
        loadModels(self)
        loadDataset(self)
        
        if self.distillType=="rd":
            self.optimizer = torch.optim.Adam(list(self.student.parameters())+list(self.bn.parameters()), lr=self.lr, betas=(0.5, 0.999)) 
        elif self.distillType=="mixed" :
            self.optimizer = torch.optim.Adam(list(self.student.parameters())+list(self.student2.parameters()), lr=self.lr, betas=(0.5, 0.999))
        else:
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr, betas=(0.5, 0.999)) 
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.lr*10,epochs=self.num_epochs,steps_per_epoch=len(self.train_loader))
        

    def train(self):
        print("training " + self.obj)
        self.student.train() 
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        
        for _ in range(1, self.num_epochs + 1):
            losses = AverageMeter()
            for sample in self.train_loader:
                image = sample['imageBase'].to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):

                    features_s,features_t  = infer(self,image) 
                    loss=cal_loss(features_s, features_t,trainer.norm)
                    losses.update(loss.sum().item(), image.size(0))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()
            
            val_loss = self.val(epoch_bar)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        epoch_bar.close()
        
        print("Training end.")

    def val(self, epoch_bar):
        self.student.eval()
        losses = AverageMeter()
        for sample in self.val_loader: 
            image = sample['imageBase'].to(self.device)
            with torch.set_grad_enabled(False):
                
                features_s,features_t  = infer(self,image)  

                loss=cal_loss(features_s, features_t,trainer.norm)
                
                losses.update(loss.item(), image.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.student.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "student.pth"))
        if self.distillType=="rd":
            state = {"model": self.bn.state_dict()}
            torch.save(state, os.path.join(self.model_dir, "bn.pth"))
        if self.distillType=="mixed":
            state = {"model": self.student2.state_dict()}
            torch.save(state, os.path.join(self.model_dir, "student2.pth"))

    
    @torch.no_grad()
    def test(self):

        self.student=loadWeights(self.student,self.model_dir,"student.pth")
        if self.distillType=="rd":
            self.bn=loadWeights(self.bn,self.model_dir,"bn.pth")
        if self.distillType=="mixed":
            self.student2=loadWeights(self.student2,self.model_dir,"student2.pth")
        
        
        kwargs = ({"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {} )
        test_dataset = MVTecDataset(
            root_dir=self.data_path+"/"+self.obj+"/test/",
            resize_shape=[self.img_resize,self.img_resize],
            crop_size=[self.img_cropsize,self.img_cropsize],
            phase='test'
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        scores = []
        test_imgs = []
        gt_list = []
        progressBar = tqdm(test_loader)
        for sample in test_loader:
            label=sample['has_anomaly']
            image = sample['imageBase'].to(self.device)
            test_imgs.extend(image.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            with torch.set_grad_enabled(False):
                
                features_s, features_t = infer(self,image)   
                
                score =cal_anomaly_maps(features_s,features_t,self.img_cropsize,trainer.norm) 
                
                progressBar.update()
                
            scores.append(score)

        progressBar.close()
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)
        img_roc_auc,_=computeAUROC(scores,gt_list,self.obj," "+self.distillType)
        
        
        return img_roc_auc
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    trainer = NetTrainer(data,device)
     
    if data['phase'] == "train":
        trainer.train()
        trainer.test()
    elif data['phase'] == "test":
        trainer.test()
    else:
        print("Phase argument must be train or test.")

