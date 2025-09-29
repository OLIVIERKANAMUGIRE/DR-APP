#******** IMPORT LIBRARIES**************
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

#****************************************************************************

class LongitudinalFundusDataset(Dataset):
    def __init__(self, root_dir, target_size=(512, 512), transform=None):
        self.root_dir = root_dir
        self.target_size = target_size  
        self.transform = transform or transforms.ToTensor()
        self.patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        visit1_path = os.path.join(patient_dir, "visit_1.png")
        visit2_path = os.path.join(patient_dir, "visit_2.png")

        # Load and preprocess with CLAHE
        visit1_img = self._load_and_preprocess(visit1_path)  
        visit2_img = self._load_and_preprocess(visit2_path)

        # Convert np.ndarray (H, W, C)
        visit1_img = Image.fromarray((visit1_img * 255).astype(np.uint8))
        visit2_img = Image.fromarray((visit2_img * 255).astype(np.uint8))

        if self.transform:
            visit1_img = self.transform(visit1_img)
            visit2_img = self.transform(visit2_img)

        return visit1_img, visit2_img

    def _load_and_preprocess(self, image_path):
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize according to the target size which is 512 by 512
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]))

        #  CLAHE per channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = [clahe.apply(img[:, :, i]) for i in range(3)]
        enhanced = np.stack(channels, axis=-1).astype(np.float32) / 255.0  
        return enhanced
    
    

