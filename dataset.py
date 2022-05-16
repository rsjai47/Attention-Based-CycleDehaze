import os
import sys
import torch.utils.data as data
import torchvision.transforms as tfs
import numpy as np
import torch
import random
from PIL import Image
from option import opt
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

BS = opt.bs
print(f"batch_size :{BS}")
if opt.crop:
    crop_size = opt.crop_size


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def tensorShow(tensor, titles=None):
    fig = plt.figure()
    img = make_grid(tensor)
    npimg = img.numpy()
    ax = fig.add_subplot(211)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(titles)
    plt.show()


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format=".png"):
        super(RESIDE_Dataset, self).__init__()
        print("loading dataset")
        self.size = size
        print("crop size", size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, "hazy"))
        self.haze_imgs = [os.path.join(path, "hazy", img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, "clear")
        self.transforms = self.get_transforms()
        self.transformstest = self.get_transforms_test()

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split("\\")[-1].split("_")[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        if self.train:
            hazeimage = self.transforms(haze.convert("RGB"))
            clearimage = self.transforms(clear.convert("RGB"))
        else:
            hazeimage = self.transformstest(haze.convert("RGB"))
            clearimage = self.transformstest(clear.convert("RGB"))

        return hazeimage, clearimage

    def get_transforms(self, resize_to=350, interpolation=Image.BICUBIC):
        all_transforms = []
        crop_s = opt.crop_size
        all_transforms.append(
            tfs.Resize(size=(resize_to, resize_to), interpolation=interpolation)
        )
        all_transforms.append(tfs.RandomCrop(crop_s))
        all_transforms.append(tfs.ToTensor())
        all_transforms.append(tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return tfs.Compose(all_transforms)

    def get_transforms_test(self):
        all_transforms = []
        all_transforms.append(tfs.ToTensor())
        all_transforms.append(tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return tfs.Compose(all_transforms)

    def __len__(self):
        return len(self.haze_imgs)


# path='data'  #path to your 'data' folder


class Eval_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format=".png"):

        super(Eval_Dataset, self).__init__()
        self.size = size
        print("crop size", size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(path)
        self.haze_imgs = [os.path.join(path, img) for img in self.haze_imgs_dir]
        self.transforms = self.get_transforms()
        self.transformstest = self.get_transforms_test()

    def __getitem__(self, index):

        haze = Image.open(self.haze_imgs[index])
        hazeimage = self.transformstest(haze.convert("RGB"))

        return hazeimage

    def get_transforms(self, resize_to=350, interpolation=Image.BICUBIC):
        all_transforms = []
        crop_s = opt.crop_size
        all_transforms.append(
            tfs.Resize(size=(resize_to, resize_to), interpolation=interpolation)
        )
        all_transforms.append(tfs.RandomCrop(crop_s))
        all_transforms.append(tfs.ToTensor())
        all_transforms.append(tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return tfs.Compose(all_transforms)

    def get_transforms_test(self, interpolation=Image.BICUBIC):
        all_transforms = []
        crop_s = 400
        all_transforms.append(tfs.Resize(size=(540, 540)))
        all_transforms.append(tfs.ToTensor())
        all_transforms.append(tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return tfs.Compose(all_transforms)

    def __len__(self):
        return len(self.haze_imgs)


pwd = os.getcwd()
print(pwd)

if __name__ == "__main__":
    loop = tqdm(ITS_train_loader, leave=True)
    for idx, (haze, clear) in enumerate(loop):
        tensorShow(haze)
        tensorShow(clear)
