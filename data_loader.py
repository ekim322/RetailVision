import glob
import os
from PIL import Image
import numpy as np
import pandas as pd
from cv2 import resize
import albumentations as A
import pyheif

import torch
from torch.utils.data import Dataset

class Alc_Dataset(Dataset):
    def __init__(self, img_dir, img_size=128, transform=None, ext='jpg', eval=False):
    # def __init__(self, img_dir, ref_dir, img_size=128, transform=None):
        img_paths = glob.glob(os.path.join(img_dir, '**/*.{}'.format(ext)), recursive=True)
        img_cats, img_cat_keys, img_classes, img_class_keys = self.extract_category_class(img_paths, img_dir)

        # ref_paths = glob.glob(os.path.join(ref_dir, '**/*.HEIC'), recursive=True)
        # ref_cats, ref_cat_keys, ref_classes, ref_class_keys = self.extract_category_class(ref_paths, ref_dir)

        transform = A.Compose([
            A.Resize(width=img_size, height=img_size),
            A.RandomCrop(width=int(img_size*0.8), height=int(img_size*0.8)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(limit=360, p=1),
            A.RandomBrightnessContrast(),
        ])

        self.img_paths = img_paths
        self.img_cats = img_cats
        self.img_cat_keys = img_cat_keys
        self.img_classes = img_classes
        self.img_class_keys = img_class_keys
        # self.ref_cats = ref_cats
        # self.ref_cat_keys = ref_cat_keys
        # self.ref_classes = ref_classes
        # self.ref_class_keys = ref_class_keys
        self.transform = transform
        self.ext = ext
        self.eval = eval

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cat = self.img_cats[idx]
        cls = self.img_classes[idx]
        
        if self.ext == 'HEIC':
            img = self.read_heif(img_path)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = (img / 255).astype(np.float32)
        tnsr_img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        if self.eval:
            return tnsr_img, torch.tensor(cat), torch.tensor(cls), img_path
        else:
            return tnsr_img, torch.tensor(cat), torch.tensor(cls)

    def extract_category_class(self, img_paths, img_dir):
        cat_strs = [img_path.split('/')[0+len(img_dir.split('/'))] for img_path in img_paths]
        unique_cat_str = np.sort(np.unique(cat_strs))
        cat_keys = {unique_cat_str[i]: range(len(unique_cat_str))[i] for i in range(len(unique_cat_str))}
        cats = np.array([cat_keys[cat_strs[i]] for i in range(len(cat_strs))])

        class_strs = [img_path.split('/')[1+len(img_dir.split('/'))] for img_path in img_paths]
        unique_class_str = np.sort(np.unique(class_strs))
        class_keys = {unique_class_str[i]: range(len(unique_class_str))[i] for i in range(len(unique_class_str))}
        classes = np.array([class_keys[class_strs[i]] for i in range(len(class_strs))])
        
        return cats, cat_keys, classes, class_keys

    def read_heif(self, path):    
        heif_file = pyheif.read(path)
        img = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        img = np.array(img)
    
        return img


