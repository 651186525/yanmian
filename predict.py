import os
import time
import json
import torch
from torchvision import transforms
from transforms import MyCrop
from PIL import Image
import numpy as np
from eva_utils.eval_metric import *
from yanMianDataset import need_horizontal_filp
import cv2
import matplotlib.pyplot as plt

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_ground_truth(json_dir, mask_root):
    target = {}
    # load json data
    json_str = open(json_dir, 'r')
    json_data = json.load(json_str)
    json_str.close()

    # get curve, landmark data
    contours_list_curve = json_data['Models']['PolygonModel2']
    curve = []
    # 去除标curve时多标的点
    for contours_list in contours_list_curve:
        if len(contours_list['Points']) > 2:
            curve.append(contours_list)
    landmark = json_data['Models']['LandMarkListModel']['Points'][0]['LabelList']
    # 将landmark转换为int，同时去除第三个维度
    landmark = {i['Label']: np.array(i['Position'], dtype=np.int32)[:2] for i in landmark}

    # get polygon mask
    mask_img = Image.open(mask_root)
    mask_array = np.array(mask_img)
    # 将mask上的像素值转为label
    mask = np.zeros(mask_array.shape, np.uint8)
    mask[mask_array == 4] = 1  # 上颌骨
    mask[mask_array == 5] = 2  # 下颌骨

    # 在mask绘制曲线  3（鼻根曲线），4（前额曲线）
    for idx in range(len(curve)):
        points = curve[idx]['Points']
        label = curve[idx]['Label']
        points_array = np.array(points, dtype=np.int32)[:, :2]
        for i in range(len(points_array) - 1):
            cv2.line(mask, points_array[i], points_array[i + 1], color=label - 3, thickness=6)
    target['mask'] = mask
    target['landmark'] = landmark

    return target


def calculate_metrics(rgb_img, gt, not_exist_landmark, is_gt: bool = True):
    landmark = gt['landmark']
    mask = gt['mask']
    img = rgb_img.copy()

    towards_right = need_horizontal_filp(Image.fromarray(img), landmark)  # 标签是否其中在图像右侧
    for j in range(1, 5):
        mask_ = np.equal(mask, j)
        img[..., 0][mask_] = 200
        img[..., 1][mask_] = 100
        img[..., 2][mask_] = 200
    for i in range(8, 14):
        cv2.circle(img, landmark[i], radius=5, color=(0, 255, 0), thickness=-1)
    # plt.imshow(rgb_img)
    # plt.show()

    upper_lip = landmark[8]
    under_lip = landmark[9]
    upper_midpoint = landmark[10]
    under_midpoint = landmark[11]
    chin = landmark[12]
    nasion = landmark[13]
    h_img = mask.shape[0]

    # 面——下部角 （IFA）  --->求不好
    angle_IFA = calculate_IFA(img, mask, 3, not_exist_landmark, nasion, chin, upper_lip, under_lip, towards_right, color=(173,255,47) )
    # 上颌 10 -鼻根 13 -下颌 11角（MNM角）   ----> 完成

    angle_MNM = calculate_MNM(img, not_exist_landmark, nasion, upper_midpoint, under_midpoint, color=(255,215,0))

    # 面——上颌角（FMA）    -----> 基本完成
    angle_FMA = calculate_FMA(img, mask, 1, not_exist_landmark, upper_lip, chin, towards_right,color=(255,106,106))

    # 颜面轮廓线（FPL） & 颜面轮廓（PL）距离     -----> 完成
    big_distance, position = calculate_PL(img, mask, 4, not_exist_landmark, under_midpoint, nasion, towards_right, color=(0,191,255))

    data = {'angle_IFA': angle_IFA, 'angle_MNM': angle_MNM, 'angle_FMA': angle_FMA, 'distance': big_distance,
            'position': position}
    plt.title('gt' if is_gt else 'pre')
    plt.imshow(img)
    plt.show()
    return data


def show_one_metric(rgb_img, gt, pre, metric: str, not_exist_landmark):
    assert metric in ['IFA', 'MNM', 'FMA', 'PL'], "metric must in ['IFA', 'MNM', 'FMA', 'PL']"
    landmark_gt = gt['landmark']
    mask_gt = gt['mask']
    landmark_pre = pre['landmark']
    mask_pre = pre['mask']

    towards_right = need_horizontal_filp(Image.fromarray(rgb_img), landmark_pre)
    towards_right2 = need_horizontal_filp(Image.fromarray(rgb_img), landmark_gt)
    assert towards_right == towards_right2, '定位偏差过大'

    upper_lip_gt = landmark_gt[8]
    under_lip_gt = landmark_gt[9]
    upper_midpoint_gt = landmark_gt[10]
    under_midpoint_gt = landmark_gt[11]
    chin_gt = landmark_gt[12]
    nasion_gt = landmark_gt[13]
    upper_lip_pre = landmark_pre[8]
    under_lip_pre = landmark_pre[9]
    upper_midpoint_pre = landmark_pre[10]
    under_midpoint_pre = landmark_pre[11]
    chin_pre = landmark_pre[12]
    nasion_pre = landmark_pre[13]
    h_img = mask_gt.shape[0]

    img = rgb_img.copy()
    if metric == 'IFA':
        # 面——下部角 （IFA）  --->求不好187 255 255
        angle_IFA = calculate_IFA(img, mask_gt, 3, not_exist_landmark, nasion_gt, chin_gt, upper_lip_gt, under_lip_gt,
                                  towards_right, color=(255, 0, 0), color_point=(255, 0, 0))
        angle_IFA = calculate_IFA(img, mask_pre, 3, not_exist_landmark, nasion_pre, chin_pre, upper_lip_pre,
                                  under_lip_pre, towards_right, color=(205, 205, 0), color_point=(205, 205, 0), color_area=(187, 255, 255))
    elif metric == 'MNM':
        # 上颌 10 -鼻根 13 -下颌 11角（MNM角）   ----> 完成
        angle_MNM = calculate_MNM(img, not_exist_landmark, nasion_gt, upper_midpoint_gt, under_midpoint_gt,
                                  color=(255, 0, 0), color_point=(255, 0, 0))
        angle_MNM = calculate_MNM(img, not_exist_landmark, nasion_pre, upper_midpoint_pre, under_midpoint_pre,
                                  color=(205, 205, 0), color_point=(205, 205, 0))
    elif metric == 'FMA':
        # 面——上颌角（FMA）    -----> 基本完成
        angle_FMA = calculate_FMA(img, mask_gt, 1, not_exist_landmark, upper_lip_gt, chin_gt, towards_right,
                                  color=(255, 0, 0), color_point=(255, 0, 0))
        angle_FMA = calculate_FMA(img, mask_pre, 1, not_exist_landmark, upper_lip_pre, chin_pre, towards_right,
                                  color=(205, 205, 0), color_point=(205, 205, 0), color_area=(187, 255, 255))
    else:
        # 颜面轮廓线（FPL） & 颜面轮廓（PL）距离     -----> 完成
        big_distance, position = calculate_PL(img, mask_gt, 4, not_exist_landmark, under_midpoint_gt, nasion_gt,
                                              towards_right, color=(255, 0, 0), color_point=(255, 0, 0))
        big_distance, position = calculate_PL(img, mask_pre, 4, not_exist_landmark, under_midpoint_pre, nasion_pre,
                                              towards_right, color=(205, 205, 0), color_point=(205, 205, 0), color_area=(187, 255, 255))
    plt.imshow(img)
    plt.title(metric)
    plt.show()


def show_predict(rgb_img, prediction, classes):
    img = rgb_img.copy()
    if prediction.shape[0] == 5 or prediction.shape[0] == 11:
        # 两个区域和两条线
        poly_curve = torch.nn.functional.softmax(prediction[:5], dim=0)
        poly_curve[poly_curve < 0.5] = 0  # 去除由于裁剪的重复阴影，同时避免小值出现
        poly_curve = np.argmax(poly_curve, axis=0)
        for j in range(1, 5):
            mask = torch.eq(poly_curve, j)
            img[..., 0][mask] = 200
            img[..., 1][mask] = 100
            img[..., 2][mask] = 200
    # plt.imshow(img)
    # plt.show()

    if prediction.shape[0] == 6 or prediction.shape[0] == 11:
        # 上唇，下唇，上颌前缘中点， 下颌前缘中点，下巴，鼻根
        for i in range(classes - 6, classes):
            keypoint = prediction[prediction.shape[0] - 6 + i]
            keypoint = np.array(keypoint.to('cpu'))
            h_shifts, w_shifts = np.where(keypoint == keypoint.max())  # np返回 行，列--》对应图片为h， w
            w_shift, h_shift = w_shifts[0], h_shifts[0] + 1
            cv2.circle(img, [w_shift, h_shift], radius=6, color=(0, 255, 0), thickness=-1)  # 点坐标x,y应为w_shift,h_shift
    plt.imshow(img)
    plt.show()


def main():
    classes = 5  # exclude background
    weights_path = "./model/heatmap/var100_Adam_6.51mse/best_model.pth"  # 127, 136
    test_txt = './data/test.txt'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_txt), f'test.txt {test_txt} not found.'
    with open(test_txt) as read:
        json_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]

    mean = (0.2371, 0.2376, 0.2380)
    std = (0.2200, 0.2202, 0.2204)
    index = {8: 'upper_lip', 9: 'under_lip', 10: 'upper_midpoint', 11: 'under_midpoint',
             12: 'chin', 13: 'nasion'}

    # get devices
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # 进入验证模式
    model.eval()

    model2 = True
    if model2:
        weights_poly_curve = './model/poly_curve/1mask_with_cross_entropy/model_149.pth'
        model_poly_curve = UNet(in_channels=3, num_classes=5, base_c=32)
        model_poly_curve.load_state_dict(torch.load(weights_poly_curve, map_location='cpu')['model'])
        model_poly_curve.to(device)
        model_poly_curve.eval()

    with torch.no_grad():  # 关闭梯度计算功能，测试不需要计算梯度
        result_pre = {'angle_IFA':[],'angle_MNM': [], 'angle_FMA': [], 'distance': [], 'position': []}
        result_gt = {'angle_IFA':[],'angle_MNM': [], 'angle_FMA': [], 'distance': [], 'position': []}
        mse = {i: [] for i in range(8, 14)}
        right_crop = False

        # 生成预测图
        for i in json_list:
            json_dir = os.path.join('./data/jsons', i)
            mask_dir = os.path.join('./data/mask', i.split('.json')[0] + '_mask.jpg')
            img_name = i.split('_jpg')[0].split('_JPG')[0]
            img_dir = os.path.join('./data/image', img_name + '.jpg')
            if not os.path.exists(img_dir):
                img_dir = os.path.join('./data/image', img_name + '.JPG')
            assert os.path.exists(img_dir), f'{img_name} does not exists.'

            original_img = Image.open(img_dir)
            img_w, img_h = original_img.size
            ground_truth = get_ground_truth(json_dir, mask_dir)

            if right_crop:
                crop_left = int(img_w * 1 / 6)
                crop_img = MyCrop(left_size=1 / 6)(original_img)
            else:
                crop_img = original_img
            crop_img = data_transform(crop_img)
            # expand batch dimension
            crop_img = torch.unsqueeze(crop_img, dim=0)


            output = model(crop_img.to(device))
            if right_crop:
                prediction_crop = output['out'].squeeze()
                prediction = torch.zeros(classes + 1, img_h, img_w)
                for crop, pre in zip(prediction_crop, prediction):
                    # 将prediction_crop复制到prediction中
                    # 使用crop_()确保了替换prediction
                    pre[:crop.shape[0], crop_left:crop.shape[1] + crop_left].copy_(crop)
            else:
                prediction = output['out'].squeeze().to('cpu')

            if model2:
                output2 = model_poly_curve(crop_img.to(device))
                if right_crop:
                    prediction_crop2 = output2['out'].squeeze()
                    prediction2 = torch.zeros(5, img_h, img_w)
                    for crop, pre in zip(prediction_crop2, prediction2):
                        # 将prediction_crop复制到prediction中
                        # 使用crop_()确保了替换prediction
                        pre[:crop.shape[0], crop_left:crop.shape[1] + crop_left].copy_(crop)
                else:
                    prediction2 = output2['out'].squeeze().to('cpu')
                prediction = torch.cat((prediction2, prediction), dim=0)

            # 生成预测数据的统一格式的target{'landmark':landmark,'mask':mask}
            pre_target, not_exist_landmark = create_target(original_img, prediction, '')
            # plt.imshow(pre_target['mask'],cmap='gray')
            # plt.show()

            # 显示预测结果
            img = np.array(original_img)
            # show_predict(img, prediction, classes + 1)
            # 计算预测数据的landmark 的 mse误差
            for i, data in enumerate(prediction[prediction.shape[0] - 6:]):
                y, x = np.where(data == data.max())
                point = ground_truth['landmark'][i + 8]  # label=i+8
                mse[i + 8].append(round(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)), 3))

            # 分指标展示
            # for metric in ['IFA', 'MNM', 'FMA', 'PL']:
            #     show_one_metric(img, ground_truth, pre_target, metric, not_exist_landmark,)

            #  计算颜面的各个指标
            pre_data = calculate_metrics(img, pre_target, not_exist_landmark, is_gt=False)
            gt_data = calculate_metrics(img, ground_truth, not_exist_landmark=[])

            for key in ['angle_IFA', 'angle_MNM', 'angle_FMA', 'distance','position']:
                result_pre[key].append(pre_data[key])
                result_gt[key].append(gt_data[key])
            a = 1
    # 评估 mse误差   var100_mse_Rightcrop最佳
    for i in range(8, 14):
        print(f'{i} :{sum(mse[i]) / len(json_list):.3f}  std: {np.std(mse[i])}')
    # # mse 50_RightCrop           100_RightCrop     var100_dice0-5_mse
    # # 8 19.75191468021172  26.29382050352002  28.257697566519354
    # # 9 18.550361388255954  9.320280322129314  10.941471138082159
    # # 10 14.49790213999321  8.55377995074959  8.179880816485136
    # # 11 39.41968865360836  22.394502893950513  22.036974424615106
    # # 12 58.48105571409088  48.89724882076784  52.64441763517149
    # # 13 13.762619819760312  6.30983834477679  13.947614695068513
    #
    # 评估颜面误差
    for i in ['angle_IFA','angle_MNM', 'angle_FMA', 'distance']:
        pre = result_pre[i]
        gt = result_gt[i]
        print(i)
        print('没有预测：', pre.count(-1))
        if pre.count(-1) >0:
            temp = pre
            temp_gt = gt
            pre = []
            gt = []
            for i in range(len(temp)):
                if temp[i] != -1:
                    pre.append(temp[i])
                    gt.append(temp_gt[i])
        print('pre平均值:', np.mean(pre))
        print('pre标准差:', np.std(pre))
        print('gt平均值:', np.mean(gt))
        print('gt标准差:', np.std(gt))
        error = []
        for m,n in zip(pre, gt):
            error.append(abs(m-n))
        print('error:', np.mean(error))
        print('error标准差:', np.std(error))
    print('position')
    print('not',result_gt['position'].count('not'))
    print('cross',result_gt['position'].count('cross'))
    print('fore',result_gt['position'].count('fore'))


if __name__ == '__main__':
    main()
