import os
import torch
import math
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
import matplotlib.pyplot as plt
from utils.util import readYamlConfig,denormalization
from utilsTraining import loadWeights,infer,cal_importance
from trainNet import NetTrainer
     
# limit the number of features to show

def visualize(trainer,layer=0,importanceSort=True,featuresToShow=1000):
    kwargs = ({"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {} )
    test_dataset = MVTecDataset(
        root_dir=trainer.data_path+"/"+trainer.obj+"/test/",
        resize_shape=[trainer.img_resize,trainer.img_resize],
        crop_size=[trainer.img_cropsize,trainer.img_cropsize],
        phase='test'
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
        
    test_imgs = []
    compteur = 0
    directory = trainer.save_path+ "/features" + "/" + trainer.obj + "/"
    os.makedirs(directory, exist_ok=True)
            
    for sample in test_loader:
        label=sample['has_anomaly']
        image = sample['imageBase'].to(trainer.device)
        imagecpu = denormalization(image.cpu().squeeze().numpy())
        test_imgs.extend(imagecpu)
        with torch.set_grad_enabled(False):
            features_s, features_t = infer(trainer,image,test=True)

            if len(features_s) <= layer:
                print("Layer index out of range, taking 0 as default index")
                layer = 0
            ft = features_t[layer].cpu().squeeze()
            fs = features_s[layer].cpu().squeeze()
                
            if importanceSort:
                ft,fs=cal_importance(ft, fs,trainer.norm) 

            plt.imshow(imagecpu)
            plt.axis('off')
            plt.savefig(directory + str(compteur) + "I.png")
            generateFeaturesImage(directory,ft, compteur,"teacher",featuresToShow,imagecpu)
            generateFeaturesImage(directory,fs, compteur,"student",featuresToShow,imagecpu)
            compteur = compteur + 1

def generateFeaturesImage(directory,input_features, compteur,type,featuresToShow,imagecpu): 
    drawnFeature=min(featuresToShow,input_features.shape[0])
    rowsColsSize = math.floor(math.sqrt(drawnFeature)) 
    
    _, axs = plt.subplots(nrows=rowsColsSize, ncols=rowsColsSize, figsize=(10, 10))
    for i in range(rowsColsSize**2):
        row = i // rowsColsSize
        col = i % rowsColsSize
        axs[row, col].imshow(input_features[i], cmap="gray")
        axs[row, col].axis("off")
    if type=="teacher":
        plt.savefig(directory + str(compteur) + "T.png")
    else:
        plt.savefig(directory + str(compteur) + "S.png")
    plt.close()
    
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    trainer = NetTrainer(data,device)
    trainer.student=loadWeights(trainer.student,trainer.model_dir,"student.pth")
    visualize(trainer,layer=0,importanceSort=True)