# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter, RandomRotation  # <-- ADDED RandomRotation here
from src.data_io import functional as F

class Compose(object):
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms: img = t(img)
        return img

class ToTensor(object):
    def __call__(self, pic): return F.to_tensor(pic)

class ToPILImage(object):
    def __call__(self, pic): return F.to_pil_image(pic)

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0)): self.size = size; self.scale = scale
    def __call__(self, img): return img.resize(self.size, Image.BILINEAR)

class RandomHorizontalFlip(object):
    def __call__(self, img):
        if np.random.rand() < 0.5: return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class Normalize(object):
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, tensor): return F.normalize(tensor, self.mean, self.std)

class SDKTestTransform(object):
    def __init__(self): self.transform = Compose([ToTensor()])
    def __call__(self, img): return self.transform(img)
