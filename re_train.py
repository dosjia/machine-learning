import torch

import global_vars
from engine import training
from util import get_model_instance_segmentation


def main():
    filename = '{}.model'
    model = get_model_instance_segmentation(num_classes=global_vars.num_classes)
    checkpoint = torch.load(filename.format(str(199)))
    model.load_state_dict(checkpoint['model_state_dict'])
    # 定义优化器和学习率调度程序
    params = [p for p in model.parameters() if p.requires_grad]
    epoch = checkpoint['epoch']
    num_epochs = 1000

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    all_lost_result = checkpoint['lost']
    avg = sum(all_lost_result[-1]) / len(all_lost_result[-1])
    if avg < 0.15:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        lr_scheduler.step()
    all_lost_avg = []
    training(all_lost_avg, all_lost_result, epoch, model, num_epochs, optimizer)


if __name__ == '__main__':
    main()
