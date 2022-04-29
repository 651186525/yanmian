import os
from PIL import Image
import numpy as np

def main():
    img_channels = 3

    # 从train.txt 中读取信息
    txt_path = './train.txt'
    with open(txt_path) as read:
        train_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]

    print(len(train_list))
    # 生成所有的图片名
    img_list = []
    for name_split in [name.split('_')[:-2] for name in train_list]:
        str = './image/'
        for temp in name_split[:-1]:
            str += temp + '_'
        str += name_split[-1]
        if os.path.exists(str + '.jpg'):
            img_list.append(str + '.jpg')
        elif os.path.exists(str + '.JPG'):
            img_list.append(str + '.JPG')
        else:
            raise '{}.jpg/JPG does not exists'.format(str)

    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    height = 0
    width = 0
    height_max = 0
    width_max = 0
    for img_path in img_list:
        img = Image.open(img_path)
        w, h = img.size
        height += h
        width += w
        if w > width_max:
            width_max = w
        if h > height_max:
            height_max = h

        img = np.array(img) / 255.
        img = img.reshape(-1, 3)
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_list)
    std = cumulative_std / len(img_list)
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f'average height : {height/len(img_list)}')
    print(f'average width : {width/len(img_list)}')
    print(f'max height : {height_max}')
    print(f'max width : {width_max}')
    # mean: [0.22270182 0.22453914 0.22637838]  [0.22922716 0.22991307 0.23040238]  [0.23326538 0.23382138 0.23423279]
    # std: [0.21268971 0.21371627 0.21473691]   [0.21878482 0.21936761 0.21957956]  [0.21978455 0.22021321 0.22034053]
    # average height: 762.9803921568628     739.983651226158  741.8021978021978
    # average width: 1088.7843137254902    1040.2125340599455  1036.2717282717283
    # max height: 866
    # max width: 1260

    # data 4                                               # check 3
    # mean: [0.23419478 0.23462838 0.2349651 ]         [0.2347, 0.2350, 0.2353]
    # std: [0.22031093 0.22061494 0.2206662 ]          [0.2209, 0.2211, 0.2211]

if __name__ == '__main__':
    main()
