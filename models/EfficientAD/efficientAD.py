import torch
from models.EfficientAD.common import get_pdn_small,get_pdn_medium,get_autoencoder

def loadPdnTeacher(trainer): 
    if trainer.modelName=="small":
        trainer.teacher = get_pdn_small().to(trainer.device)
        weights="models/EfficientAD/TeacherWeights/teacher_small.pth"
    elif trainer.modelName=="medium":
        trainer.teacher = get_pdn_medium().to(trainer.device)
        weights="models/EfficientAD/TeacherWeights/teacher_medium.pth"
    else:
        raise Exception("Invalid pdn model :  Choices are ['small', 'medium']")
    state_dict = torch.load(weights, map_location='cpu')
    trainer.teacher.load_state_dict(state_dict)

    