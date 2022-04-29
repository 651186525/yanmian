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
    line []: line为两个点组成的直线
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
    shift_h, shift_w = np.where(temp_mask == 255)

    # 去除鼻根右边（朝右）或左边多余的点
    index = shift_w<nasion[0] if towards_right else shift_w>nasion[0]
    if any(index):
        shift_h = shift_h[index]
        shift_w = shift_w[index]
    poly_fit = np.polyfit(shift_w, shift_h, 1)  # 用1次多项式拟合
    curve_function = np.poly1d(poly_fit)
    # 方案1 ：计算拟合曲线的平均点
    # mean_point = [np.mean(shift_w), curve_function(np.mean(shift_w))]
    # _, k, _ = get_angle_k_b(nasion, mean_point, h_img)

    # 方案2，计算拟合曲线斜率，然后将拟合曲线平移到鼻根处，然后作图
    # 计算拟合函数的斜率, 并求得当前斜率，同时过鼻根点的偏移b
    temp_point1 = curve_function(nasion[0]+10)
    temp_point2 = curve_function(nasion[0]-10)
    _, k, _ = get_angle_k_b([nasion[0]+10, temp_point1], [nasion[0]-10,temp_point2], h_img)
    b = (h_img-nasion[1]) - k*nasion[0]
    mean_point = [np.mean(shift_w), h_img-(k*np.mean(shift_w)+b)]

    keypoint = [nasion[0]+10, int(nasion[1] + 1/k*10)]
    return -1/k, nasion, [int(mean_point[0]), int(mean_point[1])], keypoint


def get_contours(mask, mask_label, h_img, towards_right=True):
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

    # method 1
    up_contours, left_point, right_point = remove_under_contours(contours, h_img)   # 处理掉下缘轮廓
    # 得到上缘轮廓上最适合用于连线的点
    m1_keypoint1, m1_keypoint2 = smallest_area_point(up_contours, left_point, right_point, h_img, towards_right)

    # method 2  找外界矩形
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    box = [[int(i[0]), int(i[1])] for i in box]
    box = sorted(box, key=lambda x:x[1])
    m2_keypoint1 = box[0]
    m2_keypoint2 = box[2] if abs(box[0][0]-box[1][0]) < abs(box[0][0]-box[2][0]) else box[1]

    box_x = sorted(box, key=lambda x:x[0])
    box_y = sorted(box, key=lambda x:x[1])
    left, right, top, bottom = box_x[0], box_x[-1], box_y[0], box_y[-1]
    d_left_top = np.linalg.norm(np.array(left)-np.array(top), 2)
    d_right_top = np.linalg.norm(np.array(right) - np.array(top), 2)
    m2_keypoint1 = top
    m2_keypoint2 = left if d_left_top > d_right_top else right

    # # method 3 得到上边缘从左到右中间1/3 距离的线段
    # shift_w, shift_h = [], []
    # l_p = (right_point[0]-left_point[0]) / 3 + left_point[0]
    # r_p = (right_point[0] - left_point[0]) / 3 * 2 + left_point[0]
    # for x, y in up_contours:
    #     if x>l_p and x < r_p:
    #         shift_h.append(y)
    #         shift_w.append(x)
    # poly_fit = np.polyfit(shift_w, shift_h, 1)  # 用1次多项式拟合
    # curve_function = np.poly1d(poly_fit)
    # ppp1 = [l_p, curve_function(l_p)]
    # ppp2 = [r_p, curve_function(r_p)]
    #
    # # 使用求得的最适合连线的点优化轮廓, 去除下缘过于突出的点
    # temp_contour = contours.reshape(contours.shape[0], -1)
    # contours_2 = []
    # distances = []
    # for i in temp_contour:
    #     distance = abs(get_distance([ppp1, ppp2], i, h_img)[0])
    #     distances.append(distance)
    # max_distance = max(distances)
    # for i, distance in zip(temp_contour, distances):
    #     if distance < max_distance * 1/10:
    #         contours_2.append(i)
    # print('distance:', max(distances))
    # contours = np.array(contours_2).reshape(len(contours_2), 1, -1)
    # rect = cv2.minAreaRect(contours)
    # box = cv2.boxPoints(rect)
    # box = [[int(i[0]), int(i[1])] for i in box]
    # box = sorted(box, key=lambda x:x[1])
    # m3_keypoint1 = box[0]
    # m3_keypoint2 = box[2] if abs(box[0][0]-box[1][0]) < abs(box[0][0]-box[2][0]) else box[1]

    return m2_keypoint1, m2_keypoint2

