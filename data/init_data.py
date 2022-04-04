import os
import random
import SimpleITK as sitk
import numpy as np
import cv2
import collections

# 随机将整个数据集划分，将信息写入txt文件中
def split_data(files_path):
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    random.seed(0)  # 设置随机种子，保证随机结果可复现
    val_rate = 0.2
    test_ = False

    # 获取所有json名称
    files_name = sorted([file for file in os.listdir(files_path) if file.endswith('.json')])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    if test_ is True:
        test_index = random.sample(range(0, len(val_index)), k=int(len(val_index) * val_rate))
        test_files = []
    for index, file_name in enumerate(files_name):
        if test_ is True and index in test_index:
            test_files.append(file_name)
        elif index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        if test_ is True:
            test_f = open('test.txt', 'x')
            test_f.write('\n'.join(test_files))
    except FileExistsError as e:
        print(e)
        exit(1)

def nii_to_mask(nii_path=None, img_save_name=None, binary_TF=False):
    itk_img = sitk.ReadImage(nii_path)
    if binary_TF:
        itk_img = sitk.Cast(sitk.RescaleIntensity(itk_img), sitk.sitkUInt8)  # 转换成0-255的二值灰度图
    img_array = sitk.GetArrayFromImage(itk_img)
    # 二值化后，img_array的取值为0和255；
    # 未二值化前，img_array的取值为0、45和46（其中45和46分别为图中两种不同标签标签的类别id值）。
    cv2.imwrite(img_save_name, img_array)


if __name__ == '__main__':
    root = os.getcwd()
    input_dir = os.path.join(root, 'json')
    assert os.path.exists(input_dir), '输入文件路径不存在'

    # 划分数据集
    split_data(input_dir)

    # 由nii数据转换为mask
    output_dir = os.path.join(root, 'json2_mask')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 批量转换
    input_file = [i for i in os.listdir(input_dir) if i.endswith('nii.gz')]
    for i in input_file:
        nii_path = os.path.join(input_dir, i)
        output_name = i.split('.')[0]
        nii_to_mask(nii_path=nii_path, img_save_name=os.path.join(output_dir, output_name + '_mask.jpg'), binary_TF=False)  # mask
        nii_to_mask(nii_path=nii_path, img_save_name=os.path.join(output_dir, output_name + '_mask_255.jpg'),
                binary_TF=True)  # mask二值化到255
