import os
import time
import json

import numpy as np
import torch
import cv2
import torchvision
from PIL import Image
import random
import matplotlib.pyplot as plt
from yanMianDataset import YanMianDataset
from torchvision import transforms
from detec_network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from detec_backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5, min_size=224, max_size=256)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def my_draw_boxes(img, target, isgt:bool=False):
    boxes = target['boxes']
    if isgt:
        scores = [1]*len(boxes)
        color = (255, 0, 0)
        text = 'gt'
        if len(boxes) == 0:
            print("没有检测到任何目标!")
    else:
        scores = target['scores']
        color = (0, 255, 0)
        text = 'pre'

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(img, [xmin,ymin], [xmax, ymax], color, 2)
        cv2.putText(img, text + str(round(float(scores[i]), 3)), [xmin, ymax],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=2)


def compute_iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)
    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0
    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union

def main():
    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=2)

    # load train weights
    train_weights = "./model/detec/data3_adam_no_pretrain/best_model.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # load image
    # original_img = Image.open("./test.png")
    dir = os.getcwd()
    test_dataset = YanMianDataset(dir, data_type='test')
    index = random.sample(range(0, len(test_dataset)), k=len(test_dataset))  # 返回list 取值要带上索引

    boxes_file = {}
    iou = []
    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        for i in range(len(test_dataset)):
            original_img, target = test_dataset[i]
            json_name = test_dataset.json_list[i]
            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            img_height, img_width = img.shape[-2:]

            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            # 将预测结果存入字典中
            best_score = np.argmax(predictions['scores'].to('cpu'))
            bbox = predictions['boxes'][best_score].to('cpu').numpy()
            boxes_file[json_name] = bbox

            draw_img = np.array(original_img)
            my_draw_boxes(draw_img, target, isgt=True)
            my_draw_boxes(draw_img, predictions, isgt=False)
            target_box = target['boxes'][0]
            iou_ = compute_iou(target_box, bbox)
            cv2.putText(draw_img, 'IOU:' + str(round(iou_,3)), [int(target_box[0]+10), int(target_box[1])+50],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=2)
            print(iou_)
            iou.append(iou_)
            plt.imshow(draw_img)
            plt.show()


            # 保存预测的图片结果
            # original_img.save("test_result.jpg")
            # plt.savefig('test_result.jpg')
    print(np.mean(iou))
    # 将预测结果写入json文件
    boxes_json = {i: [int(k) for k in j] for i, j in boxes_file.items()}
    with open('./data/boxes_file.json', 'w') as f:
        json.dump(boxes_json, f)

if __name__ == '__main__':
    main()
