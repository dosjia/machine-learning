import torch

import engine
import global_vars
import util


def main():
    # 构建 data loaders
    model = util.get_model_instance_segmentation(num_classes=global_vars.NUM_CLASSES)
    checkpoint = torch.load(util.filename.format(str(199)))
    model.load_state_dict(checkpoint['model_state_dict'])
    # 定义优化器和学习率调度程序
    params = [p for p in model.parameters() if p.requires_grad]
    epoch = checkpoint['epoch']

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    all_lost_result = checkpoint['lost']
    metric_logger = engine.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', engine.SmoothedValue(window_size=4))
    header = "test result:"
    with torch.no_grad():
        for images, targets in metric_logger.log_every(global_vars.TEST_LOADER, 100, header):
            images = list(image.to(global_vars.DEVICE) for image in images)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 得到模型的输出结果
            model.eval()
            output = model(images)
            for index, result in output:
                for i, score in enumerate(result['scores']):
                    if score.item() > global_vars.THRESHOLD:
                        max_index = i
                if max_index == -1:
                    print("no accountable result")
                else:
                    util.draw_pic_with_rectangle(images[index], max_index, result)


if __name__ == '__main__':
    main()
    exit()
