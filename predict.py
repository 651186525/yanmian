import os
import time
import json
import torch
from torchvision import transforms
from transforms import MyCrop
from PIL import Image
import numpy as np
from eva_utils.my_eval import *
import matplotlib.pyplot as plt
from src import UNet
from transforms import GetROI

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



def main():
    classes = 5  # exclude background
    weights_path = "./model/heatmap/var100_ROI30/best_model.pth"  # 127, 136
    test_txt = './data/test.txt'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_txt), f'test.txt {test_txt} not found.'
    with open(test_txt) as read:
        json_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]

    mean = (0.2292, 0.2299, 0.2304)
    std = (0.2188, 0.2194, 0.2196)
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

    model2 = False
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
        get_roi = True

        # 生成预测图
        for i in json_list:
            json_dir = os.path.join('./data/jsons', i)
            mask_dir = os.path.join('./data/masks', i.split('.json')[0] + '_mask.jpg')
            img_name = i.split('_jpg')[0].split('_JPG')[0]
            img_dir = os.path.join('./data/image', img_name + '.jpg')
            if not os.path.exists(img_dir):
                img_dir = os.path.join('./data/image', img_name + '.JPG')
            assert os.path.exists(img_dir), f'{img_name} does not exists.'

            original_img = Image.open(img_dir)
            ground_truth = get_ground_truth(json_dir, mask_dir)
            if get_roi:
                ROI_img, poly_curve, landmark = GetROI(border_size=30)(original_img, ground_truth['mask'], ground_truth['landmark'])
                ground_truth = {'mask':poly_curve, 'landmark':landmark}

            img_w, img_h = ROI_img.size
            
            input_img = ROI_img
            input_img = data_transform(input_img)
            # expand batch dimension
            input_img = torch.unsqueeze(input_img, dim=0)

            # model1 预测六个点
            output = model(input_img.to(device))
            prediction = output['out'].squeeze().to('cpu')
            # 显示预测结果
            img = np.array(ROI_img)
            for point in ground_truth['landmark'].values():
                cv2.circle(img, point, 6, (255,0,0), -1)
            # show_predict(img, prediction, classes + 1)
            # 计算预测数据的landmark 的 mse误差
            for i, data in enumerate(prediction[prediction.shape[0] - 6:]):
                y, x = np.where(data == data.max())
                point = ground_truth['landmark'][i + 8]  # label=i+8
                mse[i + 8].append(round(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)), 3))


            # model2 预测poly_curve
            if model2:
                output2 = model_poly_curve(input_img.to(device))
                prediction2 = output2['out'].squeeze().to('cpu')
                prediction = torch.cat((prediction2, prediction), dim=0)

                # 生成预测数据的统一格式的target{'landmark':landmark,'mask':mask}
                pre_target, not_exist_landmark = create_predict_target(original_img, prediction, '')
                plt.imshow(pre_target['mask'],cmap='gray')
                plt.show()

                # 分指标展示
                # for metric in ['IFA', 'MNM', 'FMA', 'PL']:
                #     show_one_metric(img, ground_truth, pre_target, metric, not_exist_landmark,)

                #  计算颜面的各个指标
                pre_data = calculate_metrics(img, pre_target, not_exist_landmark, is_gt=False)
                gt_data = calculate_metrics(img, ground_truth, not_exist_landmark=[])

                for key in ['angle_IFA', 'angle_MNM', 'angle_FMA', 'distance','position']:
                    result_pre[key].append(pre_data[key])
                    result_gt[key].append(gt_data[key])


    # 评估 mse误差   var100_mse_Rightcrop最佳
    for i in range(8, 14):
        print(f'{i} :{sum(mse[i]) / len(json_list):.3f}  std: {np.std(mse[i])}')
    # ROI 30
    # 8 :13.414  std: 26.76464003477134
    # 9 :11.337  std: 23.53760775669367
    # 10 :7.902  std: 5.412339243558436
    # 11 :8.779  std: 7.94814373512697
    # 12 :7.540  std: 9.274394684944388
    # 13 :5.496  std: 3.0807410802573796

    # 评估颜面误差
    if model2:
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
