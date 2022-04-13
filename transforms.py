import numpy as np
import random
from PIL import Image
import math
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    if isinstance(img,torch.Tensor):
        img_size = img.shape[-2:]
    elif isinstance(img, Image.Image):
        img_size = img.size  # .size为Image里的方法
    else:
        raise '图像类型错误'
    min_size = min(img_size)
    if min_size < size:
        ow, oh = img_size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RightCrop(object):
    def __init__(self, size=2/3):
        self.size = size

    def __call__(self, image, target):
        w, h = image.size
        image = F.crop(image, 0, 0, height=int(self.size*h), width=int(self.size*w))
        target = F.crop(target, 0, 0, height=int(self.size * h), width=int(self.size * w))
        return image, target

class BatchResize(object):
    def __init__(self, size):
        self.size = size
        self.stride = 32

    def __call__(self, image, target):
        max_size = self.max_by_axis([list(img.shape) for img in image])
        max_size[1] = int(math.ceil(max_size[1]/self.stride) * self.stride)
        max_size[2] = int(math.ceil(max_size[2] / self.stride) * self.stride)

        #
        img_batch_shape = [len(image)] + [3, max_size[1], max_size[2]]
        target_batch_shape = [len(image)] + [6, max_size[1], max_size[2]]
        # 创建shape为batch_shape且值全部为255的tensor
        batched_imgs = image[0].new_full(img_batch_shape, 255)
        batched_target = target[0].new_full(target_batch_shape, 255)
        for img, pad_img, t, pad_t in zip(image, batched_imgs, target, batched_target):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_t[: t.shape[0], : t.shape[1], : t.shape[2]].copy_(t)

        # image = F.resize(batched_imgs, [self.size,self.size])
        # target = F.resize(batched_target, [self.size,self.size])
        return image,target


    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

class MyCrop(object):  # 左右裁剪1/6 ,下裁剪1/3
    def __init__(self, left_size=1/6, right_size=1/6, bottom_size=1/3):
        self.left_size = left_size
        self.right_size = right_size
        self.bottom_size = bottom_size

    def __call__(self, img, target=None):
        img_w, img_h = img.size
        top = 0
        left = int(img_w * self.left_size)
        height = int(img_h * (1-self.bottom_size))
        width = int(img_w * (1-self.left_size-self.right_size))
        image = F.crop(img, top, left, height, width)
        if target is not None:
            target = F.crop(target, top, left, height, width)
            return image, target
        return image

class GetROI(object):
    def __init__(self, border_size=10):
        self.border_size = border_size

    def __call__(self, img, mask, landmark):
        img_w, img_h = img.size
        y,x = np.where(mask!=0)
        # 将landmark的值加入
        x,y = x.tolist(), y.tolist()
        x.extend([i[0] for i in landmark.values()])
        y.extend([i[1] for i in landmark.values()])
        left, right = min(x)-self.border_size, max(x)+self.border_size
        top, bottom = min(y)-self.border_size, max(y)+self.border_size
        left = left if left > 0 else 0
        right = right if right < img_w else img_w
        top = top if top > 0 else 0
        bottom = bottom if bottom < img_h else img_h
        height = bottom-top
        width = right-left
        img = F.crop(img, top, left, height, width)
        mask = F.crop(mask, top, left, height, width)
        landmark = {i:[j[0]-left, j[1]-top] for i,j in landmark.items()}
        return img, mask, landmark, [left, top, right, bottom]

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, origin_img, img, mask, landmark):
        width, height = img.size
        h, w = self.size
        # 先满足w
        ratio = w / width
        if height * ratio > h:
            ratio = h / height
        img = F.resize(img, [int(height * ratio), int(width * ratio)])
        mask = F.resize(mask, [int(height * ratio), int(width * ratio)])
        landmark = {i: [int(j[0] * ratio), int(j[1] * ratio)] for i, j in landmark.items()}
        # resize origin_img
        ow, oh= origin_img.size
        origin_img = F.resize(origin_img, [int(oh * ratio), int(ow * ratio)])
        return origin_img, img, mask, landmark, ratio