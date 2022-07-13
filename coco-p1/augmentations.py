from cvpods.data.transforms import ComposeTransform, ResizeShortestEdge, RandomFlip, NoOpTransform
import torchvision.transforms as transforms
from PIL import ImageFilter
import numpy as np


class GaussianBlur:
    def __init__(self, rad_range=[0.1, 2.0]):
        self.rad_range = rad_range

    def __call__(self, x):
        rad = np.random.uniform(*self.rad_range)
        x = x.filter(ImageFilter.GaussianBlur(radius=rad))
        return x

class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, annotation):
        if self.p < np.random.random():
            return img, annotation
        for t in self.transforms:
            img = t(img)
        return img, annotation

class ToPILImage:
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, img, annotation=None):
        return self.transform(img), annotation

class ToNumpy:
    def __call__(self, img, annotation=None):
        return np.asarray(img), annotation

class RandomGrayscale:
    def __init__(self, p=0.2):
        self.transform = transforms.RandomGrayscale(p=p)

    def __call__(self, img, annotation=None):
        return self.transform(img), annotation

class RandCrop:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
            transforms.ToPILImage(),
        ])

    def __call__(self, img, annotation=None):
        img = self.transform(img)
        return img, annotation

class WeakAug:
    def __init__(self, short_edge_length, max_size, sample_style):
        self.transform = ComposeTransform([
            ResizeShortestEdge(short_edge_length, max_size, sample_style),
            RandomFlip()
        ])

    def __call__(self, image, annotation):
        return self.transform(image, annotation)

class StrongAug:
    def __init__(self):
        self.transform = ComposeTransform([
            ToPILImage(),
            RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            RandCrop(),
            ToNumpy()
        ])

    def __call__(self, image, annotation):
        return self.transform(image, annotation)
