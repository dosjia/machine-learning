import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torchvision import transforms
import torchvision
from obj import FastRCNNPredictor


def inverse_grayscale_to_rgb(img):
    img_shape = img.shape  # 获取图像维度信息
    if len(img_shape) == 3:  # 当前为多通道图像
        return img
    else:
        img = img.squeeze()  # 这里 squeeze 函数去掉空维度（size=1）
        original_img_shape = img_shape[1:]  # 获得指定维度除去第一维度之后的所有元素
        original_img_shape = tuple(original_img_shape)
        img = torch.from_numpy(np.tile(img.numpy(), (3, 1, 1)))  # 这里tile函数用来复制第一维的元素
        img = img.permute(1, 2, 0).view(original_img_shape)  # 这里permute函数用于调度维度顺序
        return img


def draw_pic_with_rectangle(image, max_index, result):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = image * std[:, None, None] + mean[:, None, None]
    img = inverse_grayscale_to_rgb(img)
    pil_image = transforms.ToPILImage()(img)
    for i in range(max_index + 1):
        draw = ImageDraw.Draw(pil_image)
        box = result['boxes'][i].numpy()  # Specify the box coordinates as (x1, y1, x2, y2).
        draw.rectangle(box, outline='red', width=5)
        font = ImageFont.truetype('arial.ttf', size=48)
        x1, y1, x2, y2 = result['boxes'][i]
        # 计算 box 宽度和高度
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # 计算 box 中心的坐标值
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y1) / 2.0

        # 将中心坐标值保存到元组中
        center = (center_x, center_y)
        draw.text(center, str(result['labels'][i].item()), font=font,
                  fill=(0, 0, 255))
    plt.imshow(pil_image)
    plt.show()


def get_transform(train):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Grayscale(num_output_channels=1)
    ]
    if train:
        transform_list.append(transforms.RandomRotation(degrees=15))

    return transforms.Compose(transform_list)


def get_model_instance_segmentation(num_classes):
    # 加载预训练 ResNet-50 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取原始分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 用新的头替换原有头
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

