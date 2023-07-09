import torch

import global_vars
import util
from obj import MetricLogger, SmoothedValue


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=4))
    header = 'Epoch: [{}]'.format(epoch)
    lost_result = []
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 将图片和标注数据通过GPU/CPU送入设备
        images = list(image.to(device) for image in images)
        boxes = targets['boxes']
        labels = targets['labels']
        to_targets = []
        for i in range(len(boxes)):
            d = {'boxes': boxes[i], 'labels': labels[i]}
            to_targets.append(d)
        # 将图片和标注数据传递给FasterRCNN模型
        loss_dict = model(images, to_targets)

        losses = sum(loss for loss in loss_dict.values())
        lost_result.append(losses.item())
        # 反向传播并执行梯度更新
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 记录损失值和学习率
        metric_logger.update(loss=losses.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return lost_result
    # # 学习率调整程序
    # lr_scheduler.step()


def evaluate(model, data_loader, device, epoch, optimizer, all_lost_result):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    find_result = False
    saved = False
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(image.to(device) for image in images)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # 得到模型的输出结果
            output = model(images)
            max_index = -1
            for index, result in output:
                for i, score in enumerate(result['scores']):
                    if score.item() > global_vars.THRESHOLD:
                        max_index = i
                if max_index == -1:
                    print("no accountable result")
                else:
                    util.draw_pic_with_rectangle(images[index], max_index, result)
                    if (not saved) and max_index >= len(targets[index]['labels']):
                        saved = True
                        save_model(all_lost_result, epoch, images, index, model, optimizer)

    return find_result


def save_model(all_lost_result, epoch, images, index, model, optimizer):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lost': all_lost_result
    }
    torch.save(checkpoint, "./" + str(epoch) + ".model")
    traced_script_module = torch.jit.trace(model, images[index])
    traced_script_module.save("./" + str(epoch) + ".pt")


def training(all_lost_avg, all_lost_result, epoch, model, num_epochs, optimizer):
    for i in range(epoch + 1, num_epochs):
        lost_result = train_one_epoch(model, optimizer, global_vars.TRAIN_LOADER, global_vars.DEVICE, i, print_freq=50)
        all_lost_result.append(lost_result)
        avg = sum(lost_result) / len(lost_result)
        all_lost_avg.append(avg)
        print(f"lost avg：{avg:.4f}")
        # 在测试集上评估模型，输出测试结果
        if epoch % 10 == 9:
            evaluate(model, global_vars.VALIDATE_LOADER, device=global_vars.DEVICE, optimizer=optimizer, epoch=epoch,
                     all_lost_result=all_lost_result)
