import os
import time
import yaml
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v
            
            
def readYamlConfig(configFileName):
    with open(configFileName) as f:
        data=yaml.safe_load(f)
        return data
    
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8) # was transpose

    return x


def computeAUROC(scores,gt_list,obj,name="base"):
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print(obj + " image"+str(name)+" ROCAUC: %.3f" % (img_roc_auc))
    return img_roc_auc,img_scores  


def loadWeights(model,model_dir,alias):
    try:
        checkpoint = torch.load(os.path.join(model_dir, alias),weights_only=False)
    except Exception:
        raise Exception("Check saved model path.")
    model.load_state_dict(checkpoint["model"])
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    return model