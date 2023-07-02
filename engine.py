import torch
import time
import sys
import torchvision.models.detection.mask_rcnn
import utils


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.reset()
        self.window_size = window_size

    def reset(self):
        self.values = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        print(value)
        self.values.append(value)
        self.total += value
        self.count += 1
        if len(self.values) > self.window_size:
            self.total -= self.values.pop(0)
            self.count -= 1

    def synchronize_between_processes(self):
        if not hasattr(self, 'total_'):
            self.total_ = torch.tensor(0., dtype=torch.float64,
                                       device='cuda')
            self.count_ = torch.tensor(0, device='cuda')
        t = torch.tensor([self.total, self.count], dtype=torch.float64,
                         device='cuda')
        torch.distributed.all_reduce(t)
        self.total_ += t[0].item()
        self.count_ += t[1].item()

    @property
    def median(self):
        return torch.tensor(self.values).median()

    @property
    def avg(self):
        return sum(self.values) / len(self.values) if len(self.values) > 0 else 0

    @property
    def global_avg(self):
        if torch.distributed.is_initialized():
            self.synchronize_between_processes()
            return self.total_ / self.count_
        return self.avg

    @property
    def max(self):
        return max(self.values)

    def __str__(self):
        if len(self.values) > 0:
            return '{median:.4f} ({global_avg:.4f})'.format(
                median=self.median, global_avg=self.global_avg)
        else:
            return 'NaN'


class MetricLogger:
    def __init__(self, delimiter='\t'):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return getattr(super(), attr)

    def __str__(self):
        metric_strs = []
        for name, meter in self.meters.items():
            metric_str = '{}: {}'.format(name, str(meter))
            metric_strs.append(metric_str)
        return self.delimiter.join(metric_strs)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if header is not None:
            print(header)
        _iterable = iter(iterable)
        for obj in _iterable:
            yield obj
            # if self.step % print_freq == 0:
            # 使用字符串格式化打印当前度量信息
            print_str = f"{str(self)}"
            print(print_str)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=4))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 将图片和标注数据通过GPU/CPU送入设备
        images = list(image.to(device) for image in images)
        boxes = targets['boxes']
        # boxes = torch.rand(4, 11, 4)
        # boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
        labels = targets['labels']
        # labels = torch.randint(1, 10, (4, 11))
        to_targets = []
        for i in range(len(boxes)):
            d = {'boxes': boxes[i], 'labels': labels[i]}
            to_targets.append(d)
        # 将图片和标注数据传递给FasterRCNN模型
        loss_dict = model(images, to_targets)

        losses = sum(loss for loss in loss_dict.values())

        # 反向传播并执行梯度更新
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 记录损失值和学习率
        metric_logger.update(loss=losses.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # # 学习率调整程序
    # lr_scheduler.step()


def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            # images = list(image.to(device) for image in images)
            images = images.to(device)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 得到模型的输出结果
            output = model(images)
            print(output)
            # 对输出进行后处理得到预测结果
            # post_output = [utils.apply_nms(out) for out in output]
            # for i, pred_boxes in enumerate(post_output):
            #     if len(targets[i]["boxes"]) == 0:
            #         continue
            #     correct_preds = utils.compute_matches(targets[i]["boxes"], targets[i]["labels"], pred_boxes["boxes"],
            #                                           pred_boxes["labels"])
            #     metric_logger.update(num_correct=correct_preds)

        # 计算平均准确率
        # avg_acc = metric_logger.acc_avg
        # print('Accuracy: {}'.format(avg_acc))
