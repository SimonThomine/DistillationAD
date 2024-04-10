from models.ReverseDistillation.de_resnet import (
    de_resnet18, de_resnet34, de_resnet50, de_resnet101, 
    de_resnet152, de_resnext50_32x4d, de_resnext101_32x8d,
    de_wide_resnet50_2,de_wide_resnet101_2)

from models.ReverseDistillation.bn import (BN_layer, AttnBasicBlock, AttnBottleneck)

def loadStudent(trainer):
    if trainer.modelName=="resnet18":
        trainer.student = de_resnet18().to(trainer.device)
    elif trainer.modelName=="resnet34":
        trainer.student = de_resnet34().to(trainer.device)
    elif trainer.modelName=="resnet50":
        trainer.student = de_resnet50().to(trainer.device)
    elif trainer.modelName=="resnet101":
        trainer.student = de_resnet101().to(trainer.device)
    elif trainer.modelName=="resnet152":
        trainer.student = de_resnet152().to(trainer.device)
    elif trainer.modelName=="wide_resnet50_2":
        trainer.student = de_wide_resnet50_2().to(trainer.device)
    elif trainer.modelName=="wide_resnet101_2":
        trainer.student = de_wide_resnet101_2().to(trainer.device)


def loadBottleNeck(trainer):
    if trainer.modelName=="resnet18":
        trainer.bn=BN_layer(AttnBasicBlock, 2).to(trainer.device)
    elif trainer.modelName=="resnet34":
        trainer.bn=BN_layer(AttnBasicBlock, 3).to(trainer.device)
    elif trainer.modelName=="resnet50":
        trainer.bn=BN_layer(AttnBottleneck, 3).to(trainer.device)
    elif trainer.modelName=="resnet101":
        trainer.bn=BN_layer(AttnBottleneck, 3).to(trainer.device)
    elif trainer.modelName=="resnet152":
        trainer.bn=BN_layer(AttnBottleneck, 3).to(trainer.device)
    elif trainer.modelName=="wide_resnet50_2":
        trainer.bn=BN_layer(AttnBottleneck, 3).to(trainer.device)
    elif trainer.modelName=="wide_resnet101_2":
        trainer.bn=BN_layer(AttnBottleneck, 3).to(trainer.device)