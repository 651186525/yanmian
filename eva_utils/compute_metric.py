import math
import numpy as np
import cv2
import torch.nn.functional as functional
import matplotlib.pyplot as plt


def get_angle_k_b(point1, point2, h_img):
    """
    计算两个点相连的直线，对于x轴的角度 [-90, 90]
    点的坐标系原点为左上, 需转换到左下
    """
    # 将point2放到在point1上方, (由于原点为左上，上方的y值要更小）
    if point1[1] < point2[1]:
        point1, point2 = point2, point1

    point1 = [point1[0], h_img-point1[1]]
    point2 = [point2[0], h_img-point2[1]]
    x_shift = point2[0] - point1[0]
    y_shift = point2[1] - point1[1]
    if x_shift == 0:
        return 90, None, None
    k = y_shift/x_shift

    b = point1[1] - k*point1[0]
    arc = math.atan(k)  # 斜率求弧度
    angle = math.degrees(arc)   # 由弧度转换为角度

    # 若point2 在point1 右，返回正值，（说明图片整个在左上角） ， 负值亦然
    return angle, k, b


def get_angle_keypoint(line1, line2, h_img):
    """
    求两条直线的夹角和交点
    h_img 用于转换坐标系，图像坐标系原点位于左上，求交点和夹角坐标系原点为左下
    """
    angle1,k1,b1 = get_angle_k_b(line1[0], line1[1], h_img)
    angle2,k2,b2 = get_angle_k_b(line2[0], line2[1], h_img)
    if k1 == k2:
        for i in line1:
            for j in line2:
                if i==j:
                    keypoint = i
        return 0, keypoint

    # 求交点
    if angle1 == 90:
        keypoint_x = line1[0][0]
        keypoint_y = k2 * keypoint_x + b2
    elif angle2 == 90:
        keypoint_x = line2[0][0]
        keypoint_y = k1 * keypoint_x + b1
    else:
        keypoint_x = (b2 - b1) / (k1-k2)
        keypoint_y = k1 * keypoint_x + b1

    # assert keypoint_y == k2*keypoint_x + b2
    keypoint = [int(keypoint_x), int(h_img-keypoint_y)]

    if (angle1 > 0 and angle2>0) or (angle1 < 0 and angle2 < 0):
        return abs(angle1-angle2), keypoint
    return abs(angle1) + abs(angle2), keypoint

def get_distance(line, point, h_img):
    """
    求point 到直线的距离
    h_img 用于转换坐标系，图像坐标系原点位于左上，需转换到左下进行计算
    """
    angle, k, b = get_angle_k_b(line[0], line[1], h_img)
    x, y = point[0], h_img-point[1]
    line_y = k*x + b   # 直线在x点的y值
    shift_point = y - line_y   # point相对直线在y轴上的偏移
    radians = math.radians(angle)  # 由角度制转为弧度制
    distance = shift_point * math.cos(radians)

    # 计算距离与直线的交点
    shift_line = shift_point * math.sin(radians)
    shift_x = shift_line * math.cos(radians)
    shift_y = shift_line * math.sin(radians)
    keypoint = [int(x+shift_x), int(h_img- line_y- shift_y)]

    return distance, keypoint


def get_biggest_distance(mask, mask_label, line, h_img):
    """
    求mask中属于mask_label的所有点到line的最大距离
    """
    points_y, points_x = np.where(mask == mask_label)
    if len(points_y)==0:
        return None, None, None
    big_distance = 0
    head_point = [0,0]  # mask_label上的最大点
    big_keypoint = [0,0]  # 直线上相对的最大点
    for x,y in zip(points_x, points_y):
        # 计算点（x,y）到line的距离 以及 垂点keypoint
        disance, keypoint = get_distance(line, [x,y], h_img)
        if keypoint[1] < min(points_y):
            continue
        if abs(disance) > big_distance:
            big_distance = abs(disance)
            head_point = [x,y]
            big_keypoint = keypoint
    # 额骨预测线条较宽，将连线上一定误差内的所有点的都纳入,最后平均
    head_points = []
    _, k,b = get_angle_k_b(head_point, big_keypoint, h_img)
    for x,y in zip(points_x, points_y):
        # 计算误差
        if -1< h_img-y - (x*k+b) < 1:
            head_points.append([x,y])
    head_point = np.mean(head_points, axis=0, dtype=np.int)
    return big_distance, head_point, big_keypoint


def get_closest_point(mask, mask_label, point):
    """
    求mask中属于mask_label 的所有点到 点point的最小距离的点
    """
    points_y, points_x = np.where(mask == mask_label)
    print(len(points_x), len(points_y))
    small_distance = math.pow(point[0]-points_x[0],2)+math.pow(point[1]-points_y[0],2)
    small_point = [0,0]
    for x,y in zip(points_x[1:], points_y[1:]):
        if x==point[0] and y==point[1]:
            continue
        distance = math.pow(point[0]-x,2)+math.pow(point[1]-y,2)
        if distance < small_distance:
            small_distance = distance
            small_point = [x,y]
    return small_point

def get_position(line1, line2, h_img):
    """
    判断line1，在line2的位置关系
    line1在line：前(阴性-1），后（阳性 1），或重合（0）
    """
    _,k1,_ = get_angle_k_b(line1[0], line1[1], h_img)
    _,k2,_ = get_angle_k_b(line2[0], line2[1], h_img)
    if k1 > 0 and k2 > 0:
        if k1 > k2:
            return -1
        elif k1 == k2:
            return 0
        else:
            return  1
    elif k1 < 0 and k2 < 0:
        if k1 > k2:
            return 1
        elif k1 == k2:
            return 0
        else:
            return -1
    else:
        raise '位置错误'

def remove_under_contours(contours, h_img):
    """
    去除轮廓中，位于最左点，和最右点连线下方的轮廓点
    """
    contours_list = [i[0] for i in contours]
    left, right = min(i[0] for i in contours_list), max(i[0] for i in contours_list)
    # 找到整条轮廓上的，最左边，最右边的点
    for x, y in contours_list:
        if x == left:
            left_point = [x, y]
        if x == right:
            right_point = [x, y]
    # 去除轮廓上，位于left_temp 和right_temp 连线下方的点（即为下缘轮廓）
    _, k, b = get_angle_k_b(left_point, right_point, h_img)
    up_contours = []
    for x, y in contours_list:
        if (k * x + b) < (h_img - y):
            up_contours.append([x, y])
    return up_contours, left_point, right_point

def area_under_contours(contours, left_point, right_point, point, h_img):
    """
    求得在left_point 和right_point 之间的一个点point，在轮廓contours下的面积
    面积只取轮廓之下，舍弃轮廓上
    """
    area = 0
    _,k1,b1 = get_angle_k_b(left_point, point, h_img)
    _,k2,b2 = get_angle_k_b(point, right_point, h_img)
    for x,y in contours:
        if x>left_point[0] and x<point[0]:
            distance = h_img- y - (k1*x+b1)
            if distance>0:
                area += distance
            else:
                area += abs(distance)*1/3
        elif x>point[0] and x<right_point[0]:
            distance = h_img - y - (k2*x+b2)
            if distance>0:
                area += distance
            else:
                area += abs(distance) * 1 / 3
    # todo 优化
    # 轮廓上的面积取2/3试试看效果
    return area

def smallest_area_point(contours, left_point, right_point, h_img, towards_right):
    """
    求得使left_point、right_point之间的点在轮廓contours下取最小面积的点
    """
    smallest_point = [0,0]
    smallest_area = 10000
    for x,y in contours:
        if x>left_point[0] and x<right_point[0]:
            area = area_under_contours(contours, left_point, right_point, [x,y], h_img)
            if area < smallest_area:
                smallest_area = area
                smallest_point = [x,y]
    # 使另一个坐标点平移。

    if towards_right:
        temp_x = left_point[0]+10
        keypoint = left_point
    else:
        temp_x = right_point[0]-10
        keypoint = right_point
    # temp_y = 10000
    # for x,y in contours:
    #     if (temp_x-5)<x<(temp_x+5):
    #         if y<temp_y:
    #             temp_y = y
    # keypoint = [temp_x, temp_y]
    return smallest_point, keypoint

def get_nasion_vertical_line(mask, mask_label, nasion, h_img, towards_right: bool=False):
    temp_mask = np.zeros_like(mask)
    temp_mask[mask == mask_label] = 255
    # 闭运算处理缺陷，腐蚀使曲线变细，更好处理
    kernel = np.ones((3, 3), dtype=np.uint8)
    temp_mask = temp_mask.astype(np.uint8)
    # temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算处理缺陷
    # temp_mask = cv2.erode(temp_mask, kernel, iterations=2)  # 腐蚀两次减小大小
    shift_h, shift_w = np.where(temp_mask == 255)
    # 方案1
    # 拟合曲线后求导，效果不佳
    # zz = np.polyfit(shift_w, shift_h, 3)   # 使用三阶多项式拟合
    # pp = np.poly1d(zz)   #  转换为多项式
    #
    # # 求导
    # error = 0.5
    # keypoint_IFA = [0, 0]
    # k_IFA = 1
    # for x, y in zip(shift_w, shift_h):
    #     epsilon = 1e-10
    #     daoshu = (pp(x + epsilon) - pp(x)) / epsilon
    #     _, k_temp_IFA, _ = get_angle_k_b([x, y], nasion, h_img)
    #     if abs(k_temp_IFA - daoshu) < error:
    #         keypoint_IFA = [x, y]
    #         k_IFA = k_temp_IFA
    # tttt = [keypoint_IFA[0] - 50, int(keypoint_IFA[1] + k_IFA * 50)]
    # cv2.line(rgb_img, tttt, nasion, color=(255, 0, 0), thickness=2)
    # plt.imshow(rgb_img)
    # plt.show()

    # # 方案2，取nasion左右各 5 个像素的所有值平均
    # left_IFA, right_IFA = nasion[0]-3, nasion[0]+3
    # y_left = []
    # y_right = []
    # for x,y in zip(shift_w, shift_h):
    #     if x>=(nasion[0]-5) and x<nasion[0]:
    #         y_left.append(y)
    #     elif x>nasion[0] and x<=(nasion[0]+5):
    #         y_right.append(y)
    #
    # if towards_right:
    #     if 3 < (len(y_left) - len(y_right)) < 5:
    #         y_left = []
    #         for x,y in zip(shift_w, shift_h):
    #             if x >= (nasion[0] - 4) and x < nasion[0]:
    #                 y_left.append(y)
    #     elif (len(y_left) - len(y_right)) >= 5:
    #         y_left = []
    #         for x, y in zip(shift_w, shift_h):
    #             if x >= (nasion[0] - 3) and x < nasion[0]:
    #                 y_left.append(y)
    # elif not towards_right:
    #     if 3<(len(y_right)-len(y_left))<5:
    #         y_right = []
    #         for x,y in zip(shift_w, shift_h):
    #             if x>nasion[0] and x<=(nasion[0]+4):
    #                 y_right.append(y)
    #     elif (len(y_right)-len(y_left))>=5:
    #         y_right = []
    #         for x,y in zip(shift_w, shift_h):
    #             if x>nasion[0] and x<=(nasion[0]+4):
    #                 y_right.append(y)
    #
    # left_IFA_y = sum(y_left)/len(y_left)
    # right_IFA_y = sum(y_right)/len(y_right)
    # _,k,_ = get_angle_k_b([left_IFA, left_IFA_y],[right_IFA,right_IFA_y],h_img)
    # point1 = [left_IFA-20, int(k*20+left_IFA_y)]
    # point2 = [right_IFA+20, int(-k*20+right_IFA_y)]
    #
    # keypoint = [nasion[0]+10, int(nasion[1] + 1/k*10)]
    # return -1/k, point1, point2, keypoint

    # 方案3
    # 去除鼻根右边（朝右）或左边多余的点
    index = shift_w<nasion[0] if towards_right else shift_w>nasion[0]
    if any(index):
        shift_h = shift_h[index]
        shift_w = shift_w[index]
    mean_point = [int(np.mean(shift_w)), int(np.mean(shift_h))]
    _,k,_ = get_angle_k_b(mean_point, nasion, h_img)
    point1 = [nasion[0]-20, int(k*20+nasion[1])]
    point2 = [nasion[0]+20, int(-k*20+nasion[1])]

    keypoint = [nasion[0]+10, int(nasion[1] + 1/k*10)]
    return -1/k, point1, point2, keypoint


def get_contours(mask, mask_label, h_img):
    binary = np.zeros_like(mask)
    binary[mask == mask_label] = 255  # 边缘检测需要二值图像
    # cv2.CHAIN_APPROX_NONE   cv2.CHAIN_APPROX_SIMPLE
    binary = binary.astype(np.uint8)
    # binary = cv2.threshold(binary, 1,255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       # 检测的轮廓中有很多，提取最长的轮廓，为上颌骨轮廓
    temp = contours[0]
    for i in contours[1:]:
        if len(i) > len(temp):
            temp = i
    contours = temp
    # 处理掉下缘轮廓
    up_contours, left_point, right_point = remove_under_contours(contours, h_img)
    return contours, up_contours, left_point, right_point
