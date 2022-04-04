#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""avi2nii2dcm2json """

import os
import json
import cv2
import numpy as np
import nibabel as nib
import pytesseract
import io
import argparse
import numpy as np
import torch
import torch.utils.data
import cv2
from Networks.resnet_rpn import Resnet16
import glob
import os
import ipdb
import json

frame_extract_freq = 3
# original avi file folder containing all the original avi files
path = r'./avi_ori_256'
# first get the nii file, for converting to dcm format file conveniently
data_list = os.listdir(path)


def main(args):
    model = Resnet16(args=args, num_classes=args.num_classes)
    model.cuda()
    # load the detection model
    model.load_state_dict(torch.load('./output_resnet16_1/checkpoint/ckpt_best_mean_IOU.pth')["model"])
    for data in data_list:
        if data[-3:] == r'avi':
            avi_path = os.path.join(path, data)
            print(avi_path)
            cap = cv2.VideoCapture(avi_path)
            # get frame num
            frames_num = np.int(cap.get(7))
            thr = 0
            # volume = np.zeros([880, 1264, frames_num/3])
            volume = np.zeros([800, 800, frames_num // frame_extract_freq])
            img_count = 0
            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret is False:
                    break
                img_count += 1
                if img_count % frame_extract_freq != 0:
                    continue
                # pre-process to make sure the following nii converting
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # crop
                gray = gray[80:880, 279:1079]
                # rotate
                gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # flip
                gray = cv2.flip(gray, 0)
                volume[:, :, thr] = gray
                thr += 1

            """evaluate the spacing info"""
            # grab the D parameters
            cap = cv2.VideoCapture(avi_path)
            ret, frame = cap.read()
            frame_text = frame[100:400, 0:200]
            frame_text = cv2.cvtColor(frame_text, cv2.COLOR_BGR2GRAY)
            _, frame_text = cv2.threshold(frame_text, 30, 255, cv2.THRESH_BINARY_INV)
            txt = pytesseract.image_to_string(frame_text)
            txt_ = txt.split('\n')
            D = txt_[8].split(' ')[-1][0:3]
            # TO ENSURE SOME ERROR CASES
            if '.' not in D:
                D = D[0] + '.' + D[1]
            frame800 = frame[80:880, 279:1079]
            gray = cv2.cvtColor(frame800, cv2.COLOR_BGR2GRAY)

            cap.release()
            cv2.destroyAllWindows()

            # get the right part of the fig
            im_r = gray[:, 650:]
            for i in range(0, 800):
                if np.sum(im_r[i, :]) != 0:
                    up = i
                    break

            for j in range(799, -1, -1):
                if np.sum(im_r[j, :]) != 0:
                    down = j
                    break
            pixel_length = down - up
            # get the spacing
            ev_spacing = eval(D) * 10 / pixel_length
            # affine = np.array([[ev_spacing, 0., 0., 0.],
            #                    [0., ev_spacing, 0., 0.],
            #                    [0., 0., ev_spacing, 0.],
            #                    [0., 0., 0., 1.]])

            """convert to nii format"""
            # avi_nii = nib.Nifti1Image(volume.astype(np.uint8), affine=affine)
            #
            # os.makedirs(r'./nii_256', exist_ok=True)
            # save_path = os.path.join(r'./nii_256', data[:-4]+r'.nii.gz')
            # nib.save(avi_nii, save_path)

            """gen the pred json format file"""

            json_file = {}
            json_file["Config"] = {
                "Ambient": 0.3,
                "CanRepeatLabel": False,
                "Diffuse": 0.6,
                "ExcludeFileSuffixList": "",
                "FileTagTitle1": "",
                "FileTagTitle2": "",
                "FileTagTitle3": "",
                "IsLoadDicomToVideo": False,
                "IsSaveSrcFile": True,
                "KEY": "eiV2PnHe8mYoCH6nouBTnQ==",
                "LabelSavePath": "",
                "LandMark3DScale": 1.0,
                "LandMarkActorScale": 0.6,
                "PlayInterval": 100,
                "PolyPointScale": 1.0,
                "Record_Time": 5,
                "RectActorScale": 0.75,
                "SegModelPaths": None,
                "SliceCount": 1,
                "SliceSpacing": 1.0,
                "Specular": 0.1,
                "zh_CN": {
                    "Ambient": "环境光系数",
                    "CanRepeatLabel": "可重复标记",
                    "Diffuse": "漫反射系数",
                    "ExcludeFileSuffixList": "文件夹排除后缀",
                    "FileTagTitle1": "父Tag名称",
                    "FileTagTitle2": "附加Tag名称",
                    "FileTagTitle3": "子Tag名称",
                    "IsLoadDicomToVideo": "加载DICOM为视频",
                    "IsSaveSrcFile": "是否保存源文件",
                    "KEY": "加密码",
                    "LandMark3DScale": "3D场景点缩放",
                    "LandMarkActorScale": "点标注缩放",
                    "PlayInterval": "播放间隔",
                    "PolyPointScale": "轮廓点缩放",
                    "RectActorScale": "框缩放",
                    "Specular": "镜面反射系数"
                }}

            json_file["CurvePointIds"] = None
            json_file["Curves"] = [{"Shapes": None, "SliceType": 0},
                                   {"Shapes": None, "SliceType": 1},
                                   {"Shapes": None, "SliceType": 2}]

            json_file["FileInfo"] = {
                "Depth": frames_num // frame_extract_freq,
                "Height": 800,
                "Name": data.replace('.avi', '.nii.gz'),
                "Width": 800
            }
            json_file["FileName"] = data.replace('.avi', '_nii_gz')
            json_file["LabelGroup"] = [
                {
                    "Childs": None,
                    "ID": 0
                },
                {
                    "Childs": [
                        2,
                        3
                    ],
                    "ID": 1,
                    "Type": "rect"
                },
                {
                    "Childs": [
                        8,
                        9
                    ],
                    "ID": 7,
                    "Type": "landmark"
                }
            ]
            json_file["Models"] = {}
            json_file["Models"]["class AngleModel * __ptr64"] = None
            json_file["Models"]["class BoundingBox3DLabelModel * __ptr64"] = None
            json_file["Models"]["class BoundingBoxLabelModel * __ptr64"] = []
            with torch.no_grad():
                model.eval()
                args.train_val_test = 'test'
                for i in range(volume.shape[2]):

                    """if u need to predict the data per-n frames, count%n==0 can be used"""
                    if i % 5 == 0:
                        # =======================data pre-process==========================#
                        img = cv2.flip(volume[:, :, i], 0)
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        # ipdb.set_trace()
                        # cv2.imwrite('./1.png',img)
                        # image1 = torch.from_numpy(cv2.imread('./1.png', 0).astype(np.float32))
                        image = torch.from_numpy(img.astype(np.float32))
                        # image = torch.from_numpy(cv2.imread(i, 0).astype(np.float32))
                        image = image.unsqueeze(0)  #
                        image = torch.cat([image, image, image], dim=0)
                        image = (image - image.mean()) / (image.std() + 1e-5)
                        image = image.unsqueeze(0)
                        # =======================data pre-process===========================#

                        # get the box and scores
                        scores, boxes = model(image.cuda(), None, None)

                        # Vascular predict, change the thresholds according to the tp/fp curves
                        # or just according to the output results and ur experiences
                        if scores[0][0].max() > 0.95:
                            box_max = boxes[0][0][0]
                            x1, y1, x2, y2 = (box_max[0] - 400) * ev_spacing, (box_max[1] - 400) * ev_spacing, \
                                             (box_max[2] - 400) * ev_spacing, (box_max[3] - 400) * ev_spacing

                            json_file["Models"]["class BoundingBoxLabelModel * __ptr64"].append(
                                {"FrameCount": i,
                                 "Label": 2,
                                 "RotateMatrix": [
                                     1.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     1.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     1.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     0.0,
                                     1.0
                                 ],
                                 "SliceType": 2,
                                 "p1": [y1.cpu().numpy().tolist(), -x1.cpu().numpy().tolist(), 0.0],
                                 "p2": [y2.cpu().numpy().tolist(), -x2.cpu().numpy().tolist(), 0.0]
                                 },
                            )

                        # plaque predict, change the thresholds according to the tp/fp curves
                        # or just according to the output results and ur experiences
                        # if scores[0][1][0].max() > 0.8:
                        #     box_max = boxes[0][1][0]
                        #     x1, y1, x2, y2 = (box_max[0]-400)*ev_spacing, (box_max[1]-400)*ev_spacing, \
                        #                      (box_max[2]-400)*ev_spacing, (box_max[3]-400)*ev_spacing
                        #     json_file["Models"]["class BoundingBoxLabelModel * __ptr64"].append(
                        #         {"FrameCount": i,
                        #          "Label": 3,
                        #          "RotateMatrix": [
                        #              1.0,
                        #              0.0,
                        #              0.0,
                        #              0.0,
                        #              0.0,
                        #              1.0,
                        #              0.0,
                        #              0.0,
                        #              0.0,
                        #              0.0,
                        #              1.0,
                        #              0.0,
                        #              0.0,
                        #              0.0,
                        #              0.0,
                        #              1.0
                        #          ],
                        #          "SliceType": 2,
                        #          "p1": [y1.cpu().numpy().tolist(), -x1.cpu().numpy().tolist(), 0.0],
                        #          "p2": [y2.cpu().numpy().tolist(), -x2.cpu().numpy().tolist(), 0.0]
                        #          },
                        #     )
            # ================================= don`t need to change ================================== #
            json_file["Models"]["class CircleModel * __ptr64"] = None
            json_file["Models"]["class ColorLabelTableModel * __ptr64"] = [
                {
                    "Color": [
                        0,
                        0,
                        255,
                        255
                    ],
                    "Desc": "object",
                    "ID": 1
                },
                {
                    "Color": [
                        22,
                        214,
                        9,
                        255
                    ],
                    "Desc": "Vascular",
                    "ID": 2
                },
                {
                    "Color": [
                        144,
                        73,
                        241,
                        255
                    ],
                    "Desc": "Plaque",
                    "ID": 3
                },
                {
                    "Color": [
                        132,
                        225,
                        108,
                        255
                    ],
                    "Desc": "measure",
                    "ID": 7
                },
                {
                    "Color": [
                        255,
                        0,
                        0,
                        255
                    ],
                    "Desc": "p1",
                    "ID": 8
                },
                {
                    "Color": [
                        0,
                        0,
                        255,
                        255
                    ],
                    "Desc": "p2",
                    "ID": 9
                }
            ]
            json_file["Models"]["class EllipseModel * __ptr64"] = None
            json_file["Models"]["class FrameLabelModel * __ptr64"] = None
            json_file["Models"]["class LabelDetailModel * __ptr64"] = None
            json_file["Models"]["class LandMarkListModel * __ptr64"] = None
            json_file["Models"]["class MaskEditRecordModel * __ptr64"] = None
            json_file["Models"]["class MeasureModel * __ptr64"] = None
            json_file["Models"]["class MeshModel * __ptr64"] = None
            json_file["Models"]["class MprPositionModel * __ptr64"] = None
            json_file["Models"]["class PolygonModel * __ptr64"] = None
            json_file["Models"]["class ResliceLabelModel * __ptr64"] = None

            json_file["Polys"] = [
                {
                    "Shapes": None,
                    "SliceType": 0
                },
                {
                    "Shapes": None,
                    "SliceType": 1
                },
                {
                    "Shapes": None,
                    "SliceType": 2
                }
            ]
            # ================================= don`t need to change ================================== #

            # create the save path of json files
            os.makedirs('./carotid_data/json_0828_4', exist_ok=True)

            save_path = os.path.join(r'./carotid_data/json_0828_4', data.replace('.avi', '_nii_gz') + r'_Label.json')
            # write
            with open(save_path, 'w', encoding='utf8') as f:
                json.dump(json_file, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPN-class Test")
    """ RPN """

    parser.add_argument('--anchor_sizes', default=((5, 16, 32, 48, 64, 72, 80, 96, 150, 200),),
                        help='anchor size')
    parser.add_argument('--anchor_aspect_ratios', default=((0.3, 0.5, 1.0, 2.0, 2.5),), help='anchor size')
    # parser.add_argument('--anchor_sizes', default=((32, 48, 64, 72, 80, 96, 128, 150),),
    #                     help='anchor size')
    # parser.add_argument('--anchor_aspect_ratios', default=((0.5, 1.0, 2.0),), help='anchor size')
    parser.add_argument('--num_anchors_per_location', default=50, help='num_anchors_per_location')
    parser.add_argument('--rpn_pre_nms_top_n_train', default=4000, help='rpn_pre_nms_top_n_train')
    parser.add_argument('--rpn_post_nms_top_n_train', default=2000, help='rpn_post_nms_top_n_train')
    parser.add_argument('--rpn_pre_nms_top_n_test', default=2000, help='rpn_pre_nms_top_n_test')
    parser.add_argument('--rpn_post_nms_top_n_test', default=1000, help='rpn_post_nms_top_n_test')
    parser.add_argument('--rpn_nms_thresh', default=0.5, help='rpn_nms_thresh')
    parser.add_argument('--rpn_fg_iou_thresh', default=0.7, help='rpn_fg_iou_thresh')
    parser.add_argument('--rpn_bg_iou_thresh', default=0.3, help='rpn_bg_iou_thresh')
    parser.add_argument('--rpn_batch_size_per_image', default=512, help='rpn_batch_size_per_image')
    parser.add_argument('--rpn_positive_fraction', default=0.5, help='rpn_positive_fraction')
    parser.add_argument('--box_min_size', default=1e-1, help='box_min_size')
    parser.add_argument('--box_cls_score_thresh', default=0.05, help='box_cls_score_thresh')
    parser.add_argument('--box_coder_weights', default=(1.0, 1.0, 1.0, 1.0), help='box_coder_weights')
    parser.add_argument('--box_detections_per_img', default=100, help='每幅图像中所有类别的最大detections数')

    """ data """

    parser.add_argument('--image_mean', default=(0, 0, 0), help='image_mean')
    parser.add_argument('--image_std', default=(1, 1, 1), help='image_std')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    """ model """
    parser.add_argument('--num_classes', default=3, type=int, help='num classes')  # 1 、 2
    """ 训练 """
    parser.add_argument('--train_val_test', default='train', help='train or val or test')
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # ***.pth
    parser.add_argument('--batch_size', default=1, type=int, help='images batch size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay', dest='weight_decay')
    parser.add_argument('--lr_steps', default=[6, 20, 45, 96, 160, 224, 296, 360], nargs='+', type=int,
                        help='decrease lr')
    parser.add_argument('--lr_gamma', default=0.25, type=float, help='decrease lr by a factor of lr-gamma')

    """ 输出 """
    parser.add_argument('--print_freq', default=1, type=int, help='print frequency')
    args = parser.parse_args()

    main(args)

    ipdb.set_trace()
