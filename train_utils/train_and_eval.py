import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import math
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target, mse_loss, multiclass_dice_coeff
from transforms import BatchResize

def criterion(inputs, target, num_classes: int = 2, dice: bool = True, mse:bool = False, ignore_index: int = -100):
    losses = {}

    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = 0
        # 交叉熵损失：在通道方向softmax后，根据x的值计算
        if num_classes == 2:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            # 类别越少，为了平衡，可以设置更大的权重
            loss_weight = torch.as_tensor([1.0, 2.0], device=target.device)
        elif num_classes == 5:
            temp_index = [torch.where(target==i) for i in range(5)]
            index_total = target.shape[0] * target.shape[1] * target.shape[2]
            loss_weight = torch.as_tensor([index_total/(i[0].shape[0]) for i in temp_index])
            loss_weight = [float(i/loss_weight.max()) for i in loss_weight]  #
            loss_weight = torch.as_tensor(loss_weight, device=target.device)
        else:
            loss_weight = None
        # loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)  # 函数式API
        if dice is True:
            # 针对每个类别，背景，前景都需要计算他的dice系数
            # 根据gt构建每个类别的矩阵
            dice_target = build_target(target, num_classes, ignore_index)  # B * C* H * W
            # 计算两区域和两曲线的dice
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)

            # 计算dice损失，
            # loss += dice_loss(x[:,5:,...], target[:,5:,...], multiclass=True, ignore_index=ignore_index)/2
        if mse is True:
            if ignore_index > 0:
                roi_mask = torch.ne(target, ignore_index)
                pre = x[roi_mask]
                target_ = target[roi_mask]
            loss += nn.functional.mse_loss(pre, target_)
        # 总的损失为： 整幅图像的交叉熵损失和所有类别的dice损失之和
        losses[name] = loss
    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    dice = 0
    # dice_test = 0
    mse = {i:[] for i in range(8, 14)}
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, mask = image.to(device), target['mask'].to(device)
            output = model(image)
            output = output['out']
            landmark = target['landmark'][0]

            dice_target = build_target(mask, 5)
            output = nn.functional.softmax(output, dim=1)
            dice += multiclass_dice_coeff(output, dice_target)

            # 计算mse
            # for i, data in  enumerate(output[0]):
            #     data = data.to('cpu').detach()
            #     y,x = np.where(data==data.max())
            #     point = landmark[i+8]  # label=i+8
            #     mse[i+8].append(math.sqrt(math.pow(x[0]-point[0],2)+math.pow(y[0]-point[1],2)))

    # mse2 = {}
    # for i in range(8,14):
    #     mse2[i] = sum(mse[i])/len(data_loader)
    #     print(f'{i} :{sum(mse[i])/len(data_loader):.3f}')
    return {'mse_total':mse, 'mse_classes':mse}, dice/len(data_loader)


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    # MetricLogger 度量记录器 :为了统计各项数据，通过调用来使用或显示各项指标，通过具体项目自定义的函数
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # 每次遍历一个iteration
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, mask = image.to(device), target['mask'].to(device)
        # BatchResizeC = BatchResize(480)
        # image, target = BatchResizeC(image, target)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # 计算损失
            loss = criterion(output, mask, dice=True, mse=False, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 反向传播梯度
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
