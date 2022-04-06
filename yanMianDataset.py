import json
import os
import cv2
import numpy as np
from data.init_data import check_data
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import hflip, to_pil_image
from transforms import GetROI, Resize
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
        mask = torch.as_tensor(poly_curve)
        # 得到标注的ROI区域图像

        img_w, img_h = img.size
        border_size = 30
        y,x = np.where(mask!=0)
        # 将landmark的值加入
        x,y = x.tolist(), y.tolist()
        x.extend([i[0] for i in landmark.values()])
        y.extend([i[1] for i in landmark.values()])
        left, right = min(x)-border_size, max(x)+border_size
        top, bottom = min(y)-border_size, max(y)+border_size
        boxes = [[left, top, right, bottom]]
        # 转换为Tensor
        # or_boxes = torch.as_tensor(or_boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([1], dtype=torch.int64)
        iscrowd = torch.as_tensor([0], dtype=torch.int64)
        image_id = torch.tensor([index])
        # names = torch.as_tensor(names)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        target = {}
        # target['or_boxes'] = or_boxes
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target['iscrowd'] = iscrowd
        # target['names'] = names
        target["area"] = area

        # Image，和tensor通道组织方式不一样，但是还可以使用同一个transform是因为它内部根据类型做了处理
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


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
        if int(landmarks[i][0]) > w_center:
            num += 1

    if num >= 1/2 * len(landmarks):
        return True
    return False

# from transforms import RightCrop
# d = os.getcwd()
# mydata = YanMianDataset(d, data_type='test', resize=[320,320])  # , transforms=RightCrop(2/3),resize=[256,384]
# # a,b = mydata[0]
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