import torch
import os
from PIL import Image
import xml.etree.ElementTree as ET
from torch import nn


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root  # 数据集根目录
        self.transforms = transforms  # 预处理函数（可选）

        # 所有图像文件的路径以及它们对应的 XML 文件的路径
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, 'annotations'))))

    def __getitem__(self, idx):
        # 加载图像和其对应的 XML 注释
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        annotation_path = os.path.join(self.root, 'annotations', self.annotations[idx])
        img = Image.open(img_path).convert('RGB')
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # 解析 XML 到 Python 字典
        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # 将框的坐标和类别转换为 PyTorch 的 Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.int64)
        labels = [int(label) for label in labels]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 计算目标的数量
        num_objs = len(labels)

        # Box 的形式是 [x0, y0, x1, y1]，将其改为 [x, y, w, h]
        # boxes[:, 2:] -= boxes[:, :2]

        # 将所有内容封装成一个词典
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx]),
                  'num_objs': torch.tensor([num_objs])}

        # 预处理图像（如果需要）
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


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


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        cls_logits = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        return cls_logits, bbox_preds
