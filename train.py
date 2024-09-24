

import torch
from utils.util import readYamlConfig
from models.DBFAD.trainer_dbfad import DbfadTrainer
from models.EfficientAD.trainer_ead import EadTrainer
from models.RememberingNormality.RD.trainer_rnrd import RnrdTrainer
from models.RememberingNormality.ST.trainer_rnst import RnstTrainer
from models.ReverseDistillation.trainer_rd import RdTrainer
from models.StudentTeacher.trainer_st import StTrainer
from models.StudentTeacher.trainer_mixed import MixedTrainer
from models.SingleNet.trainer_sn import SnTrainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    
    distillType = data['distillType']
    if distillType == "dbfad":
        trainer = DbfadTrainer(data,device)
    elif distillType == "ead":
        trainer = EadTrainer(data,device)
    elif distillType == "rnrd":
        trainer=RnrdTrainer(data,device)
    elif distillType == "rnst":
        trainer=RnstTrainer(data,device)
    elif distillType == "rd":
        trainer = RdTrainer(data,device)
    elif distillType == "st":
        trainer = StTrainer(data,device)
    elif distillType == "mixed":
        trainer = MixedTrainer(data,device)
    elif distillType == "sn":
        trainer = SnTrainer(data,device)
    
     
    if data['phase'] == "train":
        trainer.train()
        trainer.test()
    elif data['phase'] == "test":
        trainer.test()
    else:
        print("Phase argument must be train or test.")