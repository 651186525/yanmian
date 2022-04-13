import os
import time
import json
import torch
from torchvision import transforms
from yanMianDataset import YanMianDataset
import transforms as T
from transforms import MyCrop, Resize, GetROI
from PIL import Image
import numpy as np
from eva_utils.my_eval import *
import matplotlib.pyplot as plt
from src import UNet
from train_utils.dice_coefficient_loss import build_target, multiclass_dice_coeff


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 5  # exclude background
    weights_path = "./model/heatmap/var100_ROI30_5.947/best_model.pth"  # 127, 136
    test_txt = './data/test.txt'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(test_txt), f'test.txt {test_txt} not found.'
    with open(test_txt) as read:
        json_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]

    mean = (0.2333, 0.2338, 0.2342)
    std = (0.2198, 0.2202, 0.2203)
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
    data_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_data = YanMianDataset(os.getcwd(), transforms=data_transform, data_type='test', resize=[320,320], pre_roi=True)
    # 进入验证模式
    model.eval()

    model2 = True
    if model2:
        weights_poly_curve = './model/poly_curve/data3_ROI30_no_0.7095/best_model.pth'
        model_poly_curve = UNet(in_channels=3, num_classes=5, base_c=32)
        model_poly_curve.load_state_dict(torch.load(weights_poly_curve, map_location='cpu')['model'])
        model_poly_curve.to(device)
        model_poly_curve.eval()

    with torch.no_grad():  # 关闭梯度计算功能，测试不需要计算梯度
        result_pre = {'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []}
        result_gt = {'IFA': [], 'MNM': [], 'FMA': [], 'FPL': [], 'PL': [], 'MML': [], 'FS': []}
        mse = {i: [] for i in range(8, 14)}
        dices = []
        get_roi = True

        # 生成预测图
        for index in range(len(test_data)):
            json_dir = test_data.json_list[index]
            original_img, ROI_img, ROI_target, box = test_data[index]
            towards_right = test_data.towards_right
            # expand batch dimension
            ROI_img = torch.unsqueeze(ROI_img, dim=0)

            # model1 预测六个点
            output = model(ROI_img.to(device))
            prediction = output['out'].squeeze().to('cpu')
            # 显示预测结果  --->ROI_img 为标准化的图，用于显示已经不好了
            # img = np.array(ROI_img)
            # for point in ROI_target['landmark'].values():
            #     cv2.circle(img, point, 6, (255,0,0), -1)
            # show_predict(img, prediction, classes + 1)
            # 计算预测数据的landmark 的 mse误差
            for i, data in enumerate(prediction[prediction.shape[0] - 6:]):
                y, x = np.where(data == data.max())
                point = ROI_target['landmark'][i + 8]  # label=i+8
                mse[i + 8].append(round(math.sqrt(math.pow(x[0] - point[0], 2) + math.pow(y[0] - point[1], 2)), 3))

            # model2 预测poly_curve
            if model2:
                output2 = model_poly_curve(ROI_img.to(device))
                prediction2 = output2['out'].to('cpu')
                gt_mask = torch.as_tensor(ROI_target['mask'], dtype=torch.int64).unsqueeze(0)  # unsqueeze统一格式(B,C,H,W)
                dice_target = build_target(gt_mask, 5)
                dice = multiclass_dice_coeff(torch.nn.functional.softmax(prediction2, dim=1), dice_target)
                dices.append(dice)
                print(f'dice:{dice:.3f}')
                prediction = torch.cat((prediction2.squeeze(), prediction), dim=0)

                # 生成预测数据的统一格式的target{'landmark':landmark,'mask':mask}
                pre_ROI_target, not_exist_landmark = create_predict_target(ROI_img, prediction, json_dir,
                                                                           towards_right=towards_right, deal_pre=True)
                # 将ROI target 转换为原图大小
                target = create_origin_target(ROI_target, box, original_img.size)
                pre_target = create_origin_target(pre_ROI_target, box, original_img.size)

                # plt.imshow(pre_target['mask'],cmap='gray')
                # plt.show()

                if len(not_exist_landmark) > 0:
                    print(not_exist_landmark)
                    img = np.array(original_img)
                    for j in range(1, 5):
                        mask_ = torch.where(pre_target['mask'] == j)
                        img[..., 0][mask_] = j * 15
                        img[..., 1][mask_] = j * 30 + 50
                        img[..., 2][mask_] = j * 15
                    plt.imshow(img)
                    plt.show()
                # 分指标展示
                # for metric in ['IFA', 'MNM', 'FMA', 'PL', 'MML']:
                #     show_one_metric(original_img, target, pre_target, metric, not_exist_landmark, show_img=True)
                #  计算颜面的各个指标
                pre_data = calculate_metrics(original_img, pre_target, not_exist_landmark, is_gt=False, show_img=False,
                                             compute_MML=True)
                gt_data = calculate_metrics(original_img, target, not_exist_landmark=[], show_img=False,
                                            compute_MML=True)

                for key in ['IFA', 'MNM', 'FMA', 'FPL', 'PL', 'MML', 'FS']:
                    result_pre[key].append(pre_data[key])
                    result_gt[key].append(gt_data[key])

    # 评估 mse误差   var100_mse_Rightcrop最佳
    for i in range(8, 14):
        print(f'{i} :{sum(mse[i]) / len(json_list):.3f}  std: {np.std(mse[i])}')
    # var100ROI30 （var50不好,150也不好
    # 8 :5.052  std: 6.2149917366883125
    # 9 :5.072  std: 6.211798032693594
    # 10 :6.238  std: 5.06599041862497
    # 11 :5.073  std: 2.60110245899234
    # 12 :3.846  std: 2.5803249000511546
    # 13 :3.902  std: 2.274749031171571

    # dice 误差
    print('avg_dice:', np.mean(dices))
    # 1        1`       2        3
    # 0.631    0.636    0.675    0.685

    # 评估颜面误差
    if model2:
        for i in ['IFA', 'MNM', 'FMA', 'PL', 'FS']:
            pre = result_pre[i]
            gt = result_gt[i]
            print(i)
            # python 数组可以用count 不可以用 Ture or False作为索引， numpy 数组可以用True or False作为索引，不能用count
            print('没有预测：', pre.count(-1))
            if pre.count(-1) > 0:
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
            for m, n in zip(pre, gt):
                error.append(abs(m - n))
            print('error:', np.mean(error))
            print('error标准差:', np.std(error))
        for i in ['FPL', 'MML']:
            print(i)
            print('not  gt:', result_gt[i].count('not'), '    pre: ', result_pre[i].count('not'))
            print('阳性 1  gt:', result_gt[i].count(1), '    pre: ', result_pre[i].count(1))
            print('阴性 -1  gt:', result_gt[i].count(-1), '    pre: ', result_pre[i].count(-1))
            print('0  gt:', result_gt[i].count(0), '    pre: ', result_pre[i].count(0))

    assss= 1
if __name__ == '__main__':
    main()
