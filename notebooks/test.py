import os
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib import TaskType
import torch

torch.set_float32_matmul_precision('medium')

configs = {
"dataset_root": r"F:\Projects\anomalib\notebooks\datasets\3-5",
"outputs_path": r"F:\Projects\anomalib\notebooks\datasets\3-5\outputs",
"model_name": "Patchcore",
}

dataset_root = configs["dataset_root"]
print("dataset_root: {}".format(dataset_root))

normal_folder_path = os.path.join(configs["dataset_root"], "normal")
abnormal_folder_path = os.path.join(configs["dataset_root"], "abnormal")

normal_ouput_path = os.path.join(configs["outputs_path"], configs["model_name"] , "normal_outputs")
abnormal_output_path = os.path.join(configs["outputs_path"], configs["model_name"] , "abnormal_outputs")



from lightning.pytorch.callbacks import EarlyStopping
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.callbacks import GraphLogger
from anomalib.loggers import AnomalibMLFlowLogger
from torchvision.transforms.v2 import Resize, RandomHorizontalFlip, Compose, Normalize, ToDtype,RandomAffine,RandomPerspective, Grayscale, ToTensor, Transform
from anomalib.data.image.folder import Folder, FolderDataset

import warnings
warnings.filterwarnings("ignore")


class ExtractBChannel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 可以在这里设置一些参数（如果需要的话），例如大小，填充值等
        pass
    
    # RGB (3, H, W) 的tensor类型
    def forward(self, img):
        # print(img.shape)
        # b_channel = img[2:, :, :]  # 提取 B 通道（张量的第三个通道，索引为2）
        # b_channel[b_channel < 100] = 0
        return img


class MyCustomTransform(torch.nn.Module):
    def forward(self, img):  # we assume inputs are always structured like this
        print(".....................{}".format(img.shape))
        return img

if __name__ == "__main__":



    train_transform = Compose(
        [
            #Grayscale(),
            MyCustomTransform(),
            #ToTensor(),
            #ToDtype(torch.uint8, scale=True),
            Resize((256, 256)),
            ##RandomAffine(degrees=(-5, 5), translate=(0.9, 0.9),scale=(0.95, 0.95), ),
            #RandomPerspective(distortion_scale=0.1, p=0.3),
            #ToDtype(torch.float32, scale=True),  # Normalize expects float input
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #Normalize(mean=[0.485], std=[0.229]),
        ],
    )

    eval_transform = Compose(
        [
            #Grayscale(),
            #ExtractBChannel(),
            #ToTensor(),
            #ToDtype(torch.uint8, scale=True),
            Resize((256, 256)),
            # RandomAffine(degrees=(-5, 5), translate=(0.9, 0.9),scale=(0.95, 0.95), ),
            # RandomPerspective(distortion_scale=0.1, p=0.3),
            #ToDtype(torch.float32, scale=True),  # Normalize expects float input
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #Normalize(mean=[0.485], std=[0.229]),
        ],
    )



    folder_datamodule = Folder(
        name="3-5",
        root=dataset_root,
        normal_dir="normal", abnormal_dir="abnormal",
        task=TaskType.CLASSIFICATION,
        num_workers=2,
        # image_size=(256, 256),
        train_batch_size = 2, eval_batch_size = 8,                     # 计算的时候会使用cuda，因此需要限制BS不适用默认值32；
        train_transform=train_transform, eval_transform=eval_transform,
    )

    folder_datamodule.setup()            #! 进行数据集分割


    temp = folder_datamodule.train_dataloader()

    # Train images
    i, data = next(enumerate(folder_datamodule.train_dataloader()))
    print(data.keys(), data["image"].shape)