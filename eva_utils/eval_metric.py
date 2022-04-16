import numpy as np

from eva_utils.compute_metric import *
import torch


def calculate_IFA(rgb_img, mask, mask_label, not_exist_landmark, nasion, chin, upper_lip, under_lip, towards_right,
                  color=(255, 0, 0), color_point=(0, 255, 0), color_area=(200, 100, 200)):
    if any([i == j for i in [mask_label, 8, 9, 12, 13] for j in not_exist_landmark]):
        return -1
    show_area(rgb_img, mask, mask_label, color=color_area)
    for i in [nasion, chin, upper_lip, under_lip]:
        cv2.circle(rgb_img, i, 6, color, -1)
    h_img = rgb_img.shape[0]
    # 求鼻根下缘切线的垂线斜率，p1，p2用于确定鼻根下缘切线，keypoint_IFA用于确定垂线
    nasion_vertical_k, p1, p2, keypoint_IFA = get_nasion_vertical_line(mask, mask_label, nasion, h_img, towards_right)
    # if p1 is None:
    #     return -1
    cv2.line(rgb_img, p1, p2, color=color_point, thickness=2)
    # 求得较突出唇
    a1, k1, _ = get_angle_k_b(chin, upper_lip, h_img)
    a2, k2, _ = get_angle_k_b(chin, under_lip, h_img)
    # 考虑垂直情况
    if a1 == 90:
        tuchu_point = upper_lip
    elif a2 == 90:
        tuchu_point = under_lip
    else:
        tuchu_point = upper_lip if abs(k1) > abs(k2) else under_lip
    angle_IFA, point_IFA = get_angle_keypoint([keypoint_IFA, nasion], [chin, tuchu_point], h_img)
    cv2.line(rgb_img, nasion, point_IFA, color=color, thickness=2)
    cv2.line(rgb_img, chin, point_IFA, color=color, thickness=2)
    cv2.putText(rgb_img, str(round(angle_IFA, 3)), point_IFA, cv2.FONT_HERSHEY_COMPLEX, 1.0, color=color, thickness=2)
    return angle_IFA


def calculate_MNM(rgb_img, not_exist_landmark, nasion, upper_midpoint, under_midpoint, color=(255, 0, 0),
                  color_point=(0, 255, 0)):
    if any([i == j for i in [10, 11, 13] for j in not_exist_landmark]):
        return -1
    for i in [nasion, nasion, upper_midpoint, under_midpoint]:
        cv2.circle(rgb_img, i, 6, color_point, -1)
    h_img = rgb_img.shape[0]
    angle_MNM, point_MNM = get_angle_keypoint([upper_midpoint, nasion], [under_midpoint, nasion], h_img)
    cv2.line(rgb_img, upper_midpoint, nasion, color=color, thickness=2)
    cv2.line(rgb_img, under_midpoint, nasion, color=color, thickness=2)
    cv2.putText(rgb_img, str(round(angle_MNM, 3)), point_MNM, cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
    return angle_MNM


def calculate_FMA(rgb_img, mask, mask_label, not_exist_landmark, upper_lip, chin, towards_right, color=(255, 0, 0),
                  color_point=(0, 255, 0), color_area=(200, 100, 200)):
    if any([i == j for i in [mask_label, 8, 12] for j in not_exist_landmark]):
        return -1
    show_area(rgb_img, mask, mask_label, color=color_area)
    for i in [chin, upper_lip]:
        cv2.circle(rgb_img, i, 6, color_point, -1)
    h_img = rgb_img.shape[0]
    contours, up_contours, left_point, right_point = get_contours(mask, mask_label, h_img)
    # 得到上缘轮廓上最适合用于连线的点
    # todo 改进
    jaw_keypoint, jaw_keypoint2 = smallest_area_point(up_contours, left_point, right_point, h_img, towards_right)
    angle_FMA, keypoint_FMA = get_angle_keypoint([chin, upper_lip], [jaw_keypoint, jaw_keypoint2], h_img)
    cv2.line(rgb_img, chin, keypoint_FMA, color=color, thickness=2)
    cv2.line(rgb_img, jaw_keypoint2, keypoint_FMA, color=color, thickness=2)
    cv2.putText(rgb_img, str(round(angle_FMA, 3)), keypoint_FMA, cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
    # cv2.drawContours(rgb_img, contours, contourIdx=-1, color=(0, 0, 255), thickness=3)
    return angle_FMA


def calculate_PL(rgb_img, mask, mask_label, not_exist_landmark, under_midpoint, nasion, towards_right,
                 color=(255, 0, 0), color_point=(0, 255, 0), color_area=(200, 100, 200)):
    if any([i == j for i in [mask_label, 11, 13] for j in not_exist_landmark]):
        return -1, 'not', [0,0]
    show_area(rgb_img, mask, mask_label, color=color_area)
    for i in [under_midpoint, nasion]:
        cv2.circle(rgb_img, i, 6, color_point, -1)
    h_img = rgb_img.shape[0]
    # 画曲线：cv2.line(lineType=cv2.LINE_AA)
    big_distance, big_head_point, big_line_point = get_biggest_distance(mask, mask_label, [nasion, under_midpoint],
                                                                        h_img)
    # 判断颜面轮廓线FPL与额骨的位置关系，前(阴性-1），后（阳性 1），或重合（0）
    position = 1 if big_distance > 0 else 0 if big_distance == 0 else -1
    # position = get_position([under_midpoint, nasion], [under_midpoint, big_head_point], h_img)
    cv2.line(rgb_img, under_midpoint, big_line_point, color=color, thickness=2)
    # cv2.line(rgb_img, under_midpoint, nasion, color=(255,0,0),thickness=2)
    cv2.line(rgb_img, big_head_point, big_line_point, color=color, thickness=2)
    mid_point = [int((i + j) / 2) for i, j in zip(big_line_point, big_head_point)]
    cv2.putText(rgb_img, str(round(big_distance, 3)), mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
    # cv2.putText(rgb_img, str(position), mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
    return big_distance, position, big_head_point

def calculate_MML(rgb_img, mask, mask_label, not_exist_landmark, under_midpoint, upper_midpoint, big_head_point, towards_right,
                 color=(255, 0, 0), color_point=(0, 255, 0), color_area=(200, 100, 200)):
    if any([i == j for i in [mask_label, 11, 13] for j in not_exist_landmark]):
        return -1, 'not'
    show_area(rgb_img, mask, mask_label, color=color_area)
    for i in [under_midpoint, upper_midpoint]:
        cv2.circle(rgb_img, i, 6, color_point, -1)
    h_img = rgb_img.shape[0]
    # 求得上下颌骨连线到PL求得最大距离的额骨对应点之间的距离
    distance, line_keypoint = get_distance([under_midpoint, upper_midpoint], big_head_point, h_img)
    # todo  位置该如何定量
    # 求得上下颌骨连线MML 与 下颌骨额骨点连线的位置关系 ， 前(阴性-1），后（阳性 1），或重合（0）
    position = 1 if distance > 0 else 0 if distance==0 else -1
    # position = get_position([under_midpoint, upper_midpoint], [under_midpoint, big_head_point], h_img)
    print(position)
    # 画曲线：cv2.line(lineType=cv2.LINE_AA)

    cv2.line(rgb_img, under_midpoint, line_keypoint, color=color, thickness=2)
    # cv2.line(rgb_img, under_midpoint, nasion, color=(255,0,0),thickness=2)
    cv2.line(rgb_img, big_head_point, line_keypoint, color=color, thickness=2)
    mid_point = [int((i + j) / 2) for i, j in zip(line_keypoint, big_head_point)]
    cv2.putText(rgb_img, str(round(distance, 3)), mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
    # cv2.putText(rgb_img, str(position), mid_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
    return distance, position


def show_area(img, mask, mask_label, color=(200, 100, 200)):
    mask_ = torch.where(mask==mask_label)
    img[..., 0][mask_] = color[0]
    img[..., 1][mask_] = color[1]
    img[..., 2][mask_] = color[2]
