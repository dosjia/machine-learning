from util import get_transform
from torch.utils.data import DataLoader
import torch
from obj import CustomDataset

# 数据集路径、批量大小和类别数
TRAIN_DATASET_DIR = 'D:/MachineLearning/0123456789/training'
VALIDATE_DATASET_DIR = 'D:/MachineLearning/0123456789/validation'
TEST_DATASET_DIR = 'D:/MachineLearning/0123456789/testing'

BATCH_SIZE = 1
NUM_CLASSES = 11
# 使用自定义数据集
TRAIN_DATASET = CustomDataset(TRAIN_DATASET_DIR, transforms=get_transform(True))
VALIDATE_DATASET = CustomDataset(VALIDATE_DATASET_DIR, transforms=get_transform(False))
TEST_DATASET = CustomDataset(TEST_DATASET_DIR, transforms=get_transform(False))
# 构建 data loaders
TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=1,
                          collate_fn=torch.utils.data.dataloader.default_collate)
VALIDATE_LOADER = DataLoader(VALIDATE_DATASET, batch_size=1, shuffle=False, num_workers=1,
                             collate_fn=torch.utils.data.dataloader.default_collate)
TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=True, num_workers=1,
                         collate_fn=torch.utils.data.dataloader.default_collate)
# 定义 device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

FILENAME = '{}.model'

THRESHOLD = 0.8
