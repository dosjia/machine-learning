import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
from model import get_model_instance_segmentation

import os
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.transforms.functional import to_tensor


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
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['num_objs'] = torch.tensor([num_objs])

        # 预处理图像（如果需要）
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transform = [transforms.ToTensor()]
    if train:
        transform.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform)


def main():
    # 数据集路径、批量大小和类别数
    dataset_dir = 'D:\code\machine-learning\data\learning'
    validate_set_dir = 'D:\code\machine-learning\data\\validate'
    batch_size = 1
    num_classes = 14

    # 使用自定义数据集
    transforms2 = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = CustomDataset(dataset_dir, transforms=transforms2)
    test_dataset = CustomDataset(validate_set_dir, transforms=transforms2)

    # 构建 data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              collate_fn=torch.utils.data.dataloader.default_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                             collate_fn=torch.utils.data.dataloader.default_collate)

    # 获取 Faster R-CNN 模型实例
    model = get_model_instance_segmentation(num_classes=num_classes)

    # 定义 device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 将 Faster R-CNN 模型移至 device
    model.to(device)

    # 定义优化器和学习率调度程序
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 训练模型 10 轮，并使用测试集评估每一轮的模型性能
    num_epochs = 100
    for epoch in range(num_epochs):
        # 训练一个 epoch，输出训练损失和准确率
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        # 更新学习率
        # lr_scheduler.step()
        # 在测试集上评估模型，输出测试结果
        if epoch % 10 == 0:
            evaluate(model, test_loader, device=device)


if __name__ == '__main__':
    main()
