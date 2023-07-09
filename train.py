import torch
from engine import training
from util import get_model_instance_segmentation
import global_vars


def main():
    # 获取 Faster R-CNN 模型实例
    model = get_model_instance_segmentation(num_classes=global_vars.NUM_CLASSES)

    # 将 Faster R-CNN 模型移至 device
    model.to(global_vars.DEVICE)

    # 定义优化器和学习率调度程序
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 训练模型 10 轮，并使用测试集评估每一轮的模型性能
    num_epochs = 1000
    all_lost_result = []
    all_lost_avg = []
    training(all_lost_avg, all_lost_result, 0, model, num_epochs, optimizer)


if __name__ == '__main__':
    main()
