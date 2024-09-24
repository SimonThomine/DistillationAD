import os
import time
import numpy as np
import torch
from tqdm import tqdm
from utils.mvtec import MVTecDataset
from utils.util import  AverageMeter,set_seed
from utils.functions import cal_anomaly_maps
from utils.util import computeAUROC


class BaseTrainer:          
    def __init__(self, data,device):  
        
        self.getParams(data,device)
        os.makedirs(self.model_dir, exist_ok=True)
        # You can set seed for reproducibility
        set_seed(42)
        
        self.load_model()
        self.load_data()
        self.load_optim()
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.lr*10,epochs=self.num_epochs,steps_per_epoch=len(self.train_loader))
    
    def load_optim(self):
      pass    
    
    def load_model(self):
      pass
    
    def change_mode(self,period="train"):
      pass
    
    def load_iter(self):
      pass
    
    def prepare(self):
      pass
    
    def infer(self,image,test=False):
      pass
    
    def computeLoss(self):
      pass

    def save_checkpoint(self):
      pass
    
    def load_weights(self):
      pass
    
    def post_process(self):
      pass
    
    
    def getParams(self,data,device):
      self.device = device
      self.validation_ratio = 0.2
      self.data_path = data['data_path']
      self.obj = data['obj']
      self.img_resize = data['TrainingData']['img_size']
      self.img_cropsize = data['TrainingData']['crop_size']
      self.num_epochs = data['TrainingData']['epochs']
      self.lr = data['TrainingData']['lr']
      self.batch_size = data['TrainingData']['batch_size']   
      self.save_path = data['save_path']
      self.model_dir = f'{self.save_path}/models/{self.obj}'
      self.img_dir = f'{self.save_path}/imgs/{self.obj}'  
      self.distillType=data['distillType']
      self.norm = data['TrainingData']['norm']

      # Model specific parameters
      self.modelName = data['backbone'] if 'backbone' in data else None
      self.outIndices = data['out_indice'] if 'out_indice' in data else None
      self.embedDim = data['embedDim'] if 'embedDim' in data else None
      self.lambda1 = data['lambda1'] if 'lambda1' in data else None
      self.lambda2 = data['lambda1'] if 'lambda2' in data else None
    
    
    def load_data(self):
      kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
      train_dataset = MVTecDataset(root_dir=f"{self.data_path}/{self.obj}/train/good", #self.data_path+"/"+self.obj+"/train/good"
          resize_shape=self.img_resize,
          crop_size=self.img_cropsize,
          phase='train'
      )
      img_nums = len(train_dataset)
      valid_num = int(img_nums * self.validation_ratio)
      train_num = img_nums - valid_num
      train_data, val_data = torch.utils.data.random_split(
          train_dataset, [train_num, valid_num]
      )
      self.train_loader=torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
      self.val_loader=torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)



    def train(self):
        print("training " + self.obj)
        
        self.change_mode("train")
        
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs,desc="Training",unit="batch")
        
        for _ in range(1, self.num_epochs + 1):
            losses = AverageMeter()
            
            self.load_iter()
            
            for sample in self.train_loader:
                image = sample['imageBase'].to(self.device)
                
                self.prepare()

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                  
                    self.infer(image)
                    loss=self.computeLoss()
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
        self.change_mode("eval")
        losses = AverageMeter()
        
        self.load_iter()
        
        for sample in self.val_loader: 
            image = sample['imageBase'].to(self.device)
            
            self.prepare()
            
            with torch.set_grad_enabled(False):
                
                self.infer(image)
                loss=self.computeLoss()
                
                losses.update(loss.item(), image.size(0))
        epoch_bar.set_postfix({"Loss": loss.item()})

        return losses.avg

    
    @torch.no_grad()
    def test(self):

        self.load_weights()
        self.change_mode("eval")
        
        kwargs = ({"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {} )
        test_dataset = MVTecDataset(
            root_dir=f'{self.data_path}/{self.obj}/test/', 
            resize_shape=self.img_resize,
            crop_size=self.img_cropsize,
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
                
                self.infer(image)   
                
                self.post_process()
                
                score =cal_anomaly_maps(self.features_s,self.features_t,self.img_cropsize,self.norm) 
                
                progressBar.update()
                
            scores.append(score)

        progressBar.close()
        scores = np.asarray(scores)
        gt_list = np.asarray(gt_list)
        img_roc_auc,_=computeAUROC(scores,gt_list,self.obj," "+self.distillType)
        
        return img_roc_auc
    
    




