import torchvision
from torch import nn


def get_model_instance_segmentation(num_classes):
    # 加载预训练 ResNet-50 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取原始分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 用新的头替换原有头
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


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