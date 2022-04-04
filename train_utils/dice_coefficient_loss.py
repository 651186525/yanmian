import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    # 此处， ignore_index 为255（忽略）
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # 将整个Gt，根据类别转化为ont-hot编码
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()  # [N, H, W] -> [N, H, W, C]
        # 填充回忽略区域的值 255
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)  # eq() 和 ne()
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        # 避免分母为0，此时dice应该为1
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)  # dim=1为通道维度------>得到每张特征图预测当前值的概率
    # multiclass代表针对每个类别（即通道）计算dice，（其本质还是一个个类别计算后求平均值）
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    # dice衡量两个集合的相似度，要使dice接近1，则loss定义为1-dice
    return 1 - fn(x, target, ignore_index=ignore_index)


def mse_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_mse if multiclass else mse
    return 1 - fn(x, target, ignore_index=ignore_index)


def mse(x: torch.Tensor, target: torch.Tensor, ignore_index=-100):
    batch_size = target.shape[0]
    d = 0.
    for i in range(batch_size):
        if ignore_index!=-100:
            roi_mask = torch.ne(target, ignore_index)
            x = x[roi_mask]
            target = target[roi_mask]
        d += nn.functional.mse_loss(x[i], target[i])
    return d / batch_size


def multiclass_mse(x: torch.Tensor, target: torch.Tensor, ignore_index=-100):
    loss = 0
    for channel in range(target.shape[1]):
        loss += mse(x[:,channel,...], target[:,channel,...], ignore_index)
    return loss/target.shape[1]