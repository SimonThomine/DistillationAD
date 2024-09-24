import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
import numpy as np
import glob

class MVTecDataset(Dataset):

    def __init__(self, root_dir, resize_shape=224,crop_size=224,phase="train"):
        
        assert isinstance(root_dir, str), "root_dir must be a string"
        assert isinstance(resize_shape, int), "resize_shape must be an integer"
        assert isinstance(crop_size, int), "crop_size must be an integer"
        
        
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.size=resize_shape
        self.crop_size=crop_size
        
        self.transform=T.Compose([T.CenterCrop(crop_size),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.phase=phase
        
    def __len__(self):
        if self.phase=="test":
            return len(self.images)
        else:
            return len(self.image_paths)

    def transform_image(self, image_path):
        
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, dsize=(self.size, self.size))
            
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)/ 255.0
        
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image=np.asarray(self.transform(torch.from_numpy(image)))
        
        if self.phase=="test":
            return image
        else:
            return image
 
    def __getitem__(self, idx):
        if self.phase=="test":
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_path = self.images[idx]
            dir_path, file_name = os.path.split(img_path)
            base_dir = os.path.basename(dir_path)
            image = self.transform_image(img_path)
            if base_dir == 'good':
                has_anomaly = np.array([0], dtype=np.int64)
            else:
                has_anomaly = np.array([1], dtype=np.int64)
            sample = {'imageBase': image, 'has_anomaly': has_anomaly, 'idx': idx}
        else:
            idx = torch.randint(0, len(self.image_paths), (1,)).item()
            image = self.transform_image(self.image_paths[idx])
            sample = {'imageBase': image}
        return sample