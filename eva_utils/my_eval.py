import json
import numpy as np
from PIL import Image
import cv2
import torch
from yanMianDataset import need_horizontal_filp
from eva_utils.eval_metric import *


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
    target['mask'] = torch.as_tensor(mask)
    target['landmark'] = landmark

    return target


def show_predict(rgb_img, prediction, classes):
    import torch
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

    if prediction.shape[0] == 6 or prediction.shape[0] == 11:
        # 上唇，下唇，上颌前缘中点， 下颌前缘中点，下巴，鼻根
        for i in range(classes - 6, classes):
            keypoint = prediction[prediction.shape[0] - 6 + i]
            keypoint = np.array(keypoint.to('cpu'))
            h_shifts, w_shifts = np.where(keypoint == keypoint.max())  # np返回 行，列--》对应图片为h， w
            w_shift, h_shift = w_shifts[0], h_shifts[0] + 1
            cv2.circle(img, [w_shift, h_shift], radius=6, color=(0, 255, 0), thickness=-1)  # 点坐标x,y应为w_shift,h_shift
    cv2.putText(img, 'red: gt', [20,20] ,cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
    cv2.putText(img, 'green: pre', [20, 35], cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(0,255,0), thickness=1)
    plt.imshow(img)
    plt.show()


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
                                  under_lip_pre, towards_right, color=(205, 205, 0), color_point=(205, 205, 0),
                                  color_area=(187, 255, 255))
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
                                              towards_right, color=(205, 205, 0), color_point=(205, 205, 0),
                                              color_area=(187, 255, 255))
    plt.imshow(img)
    plt.title(metric)
    plt.show()


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
    angle_IFA = calculate_IFA(img, mask, 3, not_exist_landmark, nasion, chin, upper_lip, under_lip, towards_right,
                              color=(173, 255, 47))
    # 上颌 10 -鼻根 13 -下颌 11角（MNM角）   ----> 完成

    angle_MNM = calculate_MNM(img, not_exist_landmark, nasion, upper_midpoint, under_midpoint, color=(255, 215, 0))

    # 面——上颌角（FMA）    -----> 基本完成
    angle_FMA = calculate_FMA(img, mask, 1, not_exist_landmark, upper_lip, chin, towards_right, color=(255, 106, 106))

    # 颜面轮廓线（FPL） & 颜面轮廓（PL）距离     -----> 完成
    big_distance, position = calculate_PL(img, mask, 4, not_exist_landmark, under_midpoint, nasion, towards_right,
                                          color=(0, 191, 255))

    data = {'angle_IFA': angle_IFA, 'angle_MNM': angle_MNM, 'angle_FMA': angle_FMA, 'distance': big_distance,
            'position': position}
    plt.title('gt' if is_gt else 'pre')
    plt.imshow(img)
    plt.show()
    return data
