import json
import os
import cv2
import numpy as np
from data.init_data import check_data
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import hflip, resize
from transforms import GetROI
import matplotlib.pyplot as plt


class YanMianDataset(Dataset):
    def __init__(self, root: str, transforms=None, data_type: str = 'train', resize=None):
        assert data_type in ['train', 'val', 'test'], "data_type must be in ['train', 'val', 'test']"
        self.root = os.path.join(root, 'data')
        self.transforms = transforms
        self.resize = resize
        self.json_list = []
        self.data_type = data_type

        # read txt file and save all json file list (train/val/test)
        json_path = os.path.join(self.root, 'jsons')
        txt_path = os.path.join(self.root, data_type + '.txt')
        assert os.path.exists(txt_path), 'not found {} file'.format(data_type + '.txt')
        with open(txt_path) as read:
            self.json_list = [os.path.join(json_path, line.strip())
                              for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.json_list) > 0, 'in "{}" file does not find any information'.format(data_type + '.txt')
        for json_dir in self.json_list:
            assert os.path.exists(json_dir), 'not found "{}" file'.format(json_dir)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        img_root = os.path.join(self.root, 'image')
        mask_root = os.path.join(self.root, 'masks')

        # load json data
        json_dir = self.json_list[index]
        json_str = open(json_dir, 'r')
        json_data = json.load(json_str)
        json_str.close()

        # get image
        img_name = json_data['FileInfo']['Name']
        img_path = os.path.join(img_root, img_name)
        img = Image.open(img_path)
        target = {}

        # get curve, landmark data
        temp_curve = json_data['Models']['PolygonModel2']  # list   [index]['Points']
        curve = []
        # 去除标curve时多标的点
        for temp in temp_curve:
            if len(temp['Points']) > 2:
                curve.append(temp)
        landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
        # 将landmark转换为int，同时去除第三个维度
        landmark = {i['Label'] : np.array(i['Position'], dtype=np.int32)[:2] for i in landmark}
        poly_points = json_data['Polys'][0]['Shapes']
        # get polygon mask
        mask_path = os.path.join(mask_root, json_dir.split('/')[-1].split('.')[0] + '_mask.jpg')
        # !!!!!!!!! np会改变Image的通道排列顺序，Image，为cwh，np为hwc，一般为chw（cv等等） cv:hwc
        mask_img = Image.open(mask_path)   # Image： c, w, h
        mask_array = np.array(mask_img)   # np 会将shape变为 h, w, c

        # check data
        check_data(curve, landmark, poly_points, json_dir, self.data_type)

        # 生成poly_curve 图
        poly_curve = np.zeros_like(mask_array)
        for i in range(4):
            if i==0 or i==1:  # 两个区域
                poly_curve[mask_array==i+4] = i+1
            elif i==2 or i==3:  # 两条直线
                points = curve[i-3]['Points']
                label = curve[i-3]['Label']
                points_array = np.array(points, dtype=np.int32)[:, :2]
                for j in range(len(points_array) - 1):
                    cv2.line(poly_curve, points_array[j], points_array[j + 1], color=label-3, thickness=6)
        # poly_curve = Image.fromarray(poly_curve)
        poly_curve = torch.as_tensor(poly_curve)
        # 得到标注的ROI区域图像
        img, poly_curve, landmark = GetROI(border_size=30)(img, poly_curve, landmark)

        # 生成mask
        mask = torch.zeros(6, *poly_curve.shape, dtype=torch.float)
        # 根据landmark 绘制高斯热图 （进行点分割）
        # heatmap 维度为 c,h,w 因为ToTensor会将Image(c.w,h)也变为(c,h,w)
        for label in landmark:
            point = landmark[label]
            temp_heatmap = make_2d_heatmap(point, poly_curve.shape, max_value=20, var=100)
            mask[label-8] = temp_heatmap

        # 将位于右上角的图片翻转到左上角
        # if need_horizontal_filp(img, landmark):
        #     img = hflip(img)
        #     mask = hflip(mask)
        #     landmark = {i:[width-j[0],j[1]] for i,j in landmark.items()}   # 将landmark也翻转

        # resize image
        if self.resize is not None:
            width,height = img.size
            h, w = self.resize
            # 先满足w
            ratio = w/width
            if height*ratio > h:
                ratio = h/height
            img = resize(img, [int(height*ratio), int(width*ratio)])
            mask = resize(mask, [int(height*ratio), int(width*ratio)])
            landmark = {i:[int(j[0]*ratio), int(j[1]*ratio)] for i,j in landmark.items()}

        # Image，和tensor通道组织方式不一样，但是还可以使用同一个transform是因为它内部根据类型做了处理
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # 生成target
        target['mask'] = mask
        target['landmark'] = landmark

        return img, target


    @staticmethod
    def collate_fn(batch):  # 如何取样本，实现自定义的batch输出
        images, targets = list(zip(*batch))  # batch里每个元素表示一张图片和一个gt
        batched_imgs = cat_list(images, fill_value=0)  # 统一batch的图片大小
        mask = [i['mask'] for i in targets]
        batched_targets = {'landmark':[i['landmark'] for i in targets]}
        batched_targets['mask'] = cat_list(mask, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))  # 获取每个维度的最大值
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)  # batch, c, h, w
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def make_2d_heatmap(landmark, size, max_value = 3, var=5.0):
    """
    生成一个size大小，中心在landmark处的热图
    :param max_value: 热图中心的最大值
    :param var: 生成热图的方差
    """
    height, width = size
    landmark = (landmark[1], landmark[0])
    x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing="ij")  # 一个网格有横纵两个坐标
    p = torch.stack([x, y], dim=2)
    from math import pi, sqrt
    inner_factor = -1 / (2 * var)
    outer_factor = 1 / sqrt(var * (2 * pi) ** 2)
    mean = torch.as_tensor(landmark)
    heatmap = (p - mean).pow(2).sum(dim=-1)
    heatmap = torch.exp(heatmap * inner_factor) * outer_factor

    # 将heatmap的最大值进行缩放
    heatmap = heatmap / heatmap.max()
    heatmap = heatmap * max_value
    return heatmap

def need_horizontal_filp(img, landmarks):
    """
    根据标记点位于图像的方位判断是否需要水平翻转
    若有三分之二的landmark位于图像右上则翻转
    """
    w, h = img.size
    w_center, h_center = w/2, h/2
    num = 0
    # 统计landmark中有多少个点位于图像右上角
    for i in landmarks:
        if int(landmarks[i][0]) > w_center and int(landmarks[i][1]) < h_center:
            num += 1

    if num >= 1/2 * len(landmarks):
        return True
    return False

# from transforms import RightCrop
# d = os.getcwd()
# mydata = YanMianDataset(d, data_type='test', resize=[320,320])  # , transforms=RightCrop(2/3),resize=[256,384]
# # a,b = mydata[0]
# # c =1
# for i in range(len(mydata)):
#     a,b = mydata[i]
#     print(i)


# train data 734
# val data 94
# test data 89

# 第一轮标注错误情况
# 试标 22张不能用
# 1 curve: 5, 5 landmark: 3, 上颌骨（下颌骨）未被标注（无label）:7, 存在曲线未被标注（无label）:7
# 第二轮标注错误情况
# IMG_20201021_2_55_jpg_Label.json 只标了一条线，且一条线只有一个点
# 0135877_Mengyan_Tang_20200917091142414_jpg_Label.json  未标注曲线
# 1 curve: 3, 5 landmark:6, 上颌骨（下颌骨）未被标注（无label）:17, 存在曲线未被标注（无label）:1