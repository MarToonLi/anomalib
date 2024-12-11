import os
import platform
import warnings
import numpy as np
from PIL import Image
from pathlib import Path
from anomalib import TaskType


import torch
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image, to_image
from anomalib.deploy import OpenVINOInferencer, TorchInferencer


from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm, Fastflow, Uflow, EfficientAd
from lightning.pytorch.callbacks import EarlyStopping
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.callbacks import GraphLogger
from anomalib.loggers import AnomalibMLFlowLogger


os_name = platform.system()
isLinux = True if os_name.lower() == 'linux' else False

seed = 67
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')



import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def pplot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    
    
configs = {
    "dataset_root": r"/local_data/datasets/3-5-jing" if isLinux else r"F:\Projects\anomalib\notebooks\datasets\3-5 - jing",
    "outputs_path": r"/home/projects/myprojects/anomalib_projects/outputs" if isLinux else r"F:\Projects\anomalib\notebooks\datasets\3-5 - jing\outputs",
    "model_name": "Fastflow",
}
dataset_root = configs["dataset_root"]
print("dataset_root: {}".format(dataset_root))

normal_folder_path   = os.path.join(configs["dataset_root"], "normal")
abnormal_folder_path = os.path.join(configs["dataset_root"], "abnormal")
test_folder_path     = os.path.join(configs["dataset_root"], "test")

normal_ouput_path    = os.path.join(configs["outputs_path"], configs["model_name"] , "normal_outputs")
abnormal_output_path = os.path.join(configs["outputs_path"], configs["model_name"] , "abnormal_outputs")
test_output_path     = os.path.join(configs["outputs_path"], configs["model_name"] , "test_outputs")


from lightning.pytorch.callbacks import EarlyStopping
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.callbacks import GraphLogger
from anomalib.loggers import AnomalibMLFlowLogger
from torchvision.transforms.v2 import Resize, RandomHorizontalFlip, Compose, Normalize, ToDtype,RandomAffine,RandomPerspective, Grayscale, ToTensor, Transform, GaussianBlur
from anomalib.data.image.folder import Folder, FolderDataset





class ExtractBChannel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    # RGB (N, 3, H, W) 的tensor类型
    def forward(self, img):
        
        if not isinstance(img, torch.Tensor): img = torch.Tensor(img)
        
        tmp_img = img.clone()
        if len(img.shape) == 3: tmp_img = tmp_img.unsqueeze(0)
        bs, channels, height, width = tmp_img.shape
        
        if channels == 1: tmp_img = tmp_img.repeat(1,3,1,1)
        
        b_channel = tmp_img[:, 2, :, :]                      # 提取 B 通道（张量的第三个通道，索引为2）
        b_channel[b_channel < 100/255] = 0
        # b_channel[b_channel >= 100/255] = 1                # 不能添加
        b_channel_3 = b_channel.repeat(1, 3, 1, 1)
        
        out_img = b_channel_3
        if len(img.shape) == 3: out_img = out_img.squeeze(0)
        
        #print("{} --> {} --> {} -- {};".format(img.shape, tmp_img.shape, b_channel.shape, out_img.shape))
        return out_img


train_transform = Compose(
    [
        ExtractBChannel(),                                                # 0~1之间
        #ToTensor(),
        #ToDtype(torch.uint8, scale=True),
        Resize((256, 256)),                                               # 如果resizeHW不一致，会引起fastflow报layernorm错误
        # RandomHorizontalFlip(p=0.3),                                    # 无seed, 0.90 --> 0.95
        #RandomAffine(degrees=(-5, 5), translate=(0.95, 0.95),scale=(0.95, 0.95), ),     # onnx 不支持 grid_sampler.
        #RandomPerspective(distortion_scale=0.1, p=0.3),
        #ToDtype(torch.float32, scale=True),                              # Normalize expects float input
        #Normalize(mean=[0.406, 0.406, 0.406], std=[0.225, 0.225, 0.225]),
    ],
)

eval_transform = Compose(
    [
        ExtractBChannel(),          
        # ToTensor(),
        #ToDtype(torch.uint8, scale=True),
        Resize((256, 256)),
        #RandomHorizontalFlip(p=0.3),  
        #RandomAffine(degrees=(-5, 5), translate=(0.95, 0.95),scale=(0.95, 0.95), ), 
        #RandomPerspective(distortion_scale=0.1, p=0.3),
        #ToDtype(torch.float32, scale=True), 
        #Normalize(mean=[0.406, 0.406, 0.406], std=[0.225, 0.225, 0.225]),
    ],
)



folder_datamoduleA = Folder(
    name="3-5",
    root=dataset_root,
    normal_dir="normal", abnormal_dir="normal",
    task=TaskType.CLASSIFICATION,
    num_workers=0,                                           #! in jupyter, need to be zero. and non-0 in python main;
    train_batch_size = 1, eval_batch_size = 1,              #! 计算的时候会使用cuda，因此需要限制BS不适用默认值32；
    train_transform=train_transform, eval_transform=eval_transform,
    #transform=train_transform, 
    seed = seed,
    test_split_ratio=0.02,    #! 控制正常样本，在非训练集和训练集中的数量比例
    val_split_ratio=0.98,     #! 控制剩余正常样本和异常样本，在验证集和测试集中的数量比例
)

folder_datamoduleB = Folder(
    name="3-5",
    root=dataset_root,
    normal_dir="normal", abnormal_dir="abnormal",
    task=TaskType.CLASSIFICATION,
    num_workers=0,                                           #! in jupyter, need to be zero. and non-0 in python main;
    train_batch_size = 16, eval_batch_size = 16,              #! 计算的时候会使用cuda，因此需要限制BS不适用默认值32；
    train_transform=train_transform, eval_transform=eval_transform,
    #transform=train_transform, 
    seed = seed,
    test_split_ratio=0.98,    #! 控制正常样本，在非训练集和训练集中的数量比例
    val_split_ratio=0.98,     #! 控制剩余正常样本和异常样本，在验证集和测试集中的数量比例
)

folder_datamoduleC = Folder(
    name="3-5",
    root=dataset_root,
    normal_dir="normal", abnormal_dir="test",
    task=TaskType.CLASSIFICATION,
    num_workers=0,                                           #! in jupyter, need to be zero. and non-0 in python main;
    train_batch_size = 16, eval_batch_size = 16,              #! 计算的时候会使用cuda，因此需要限制BS不适用默认值32；
    train_transform=train_transform, eval_transform=eval_transform,
    #transform=train_transform, 
    seed = seed,
    test_split_ratio=0.1,    #! 控制正常样本，在非训练集和训练集中的数量比例
    val_split_ratio=0.1,     #! 控制剩余正常样本和异常样本，在验证集和测试集中的数量比例
)

folder_datamoduleA.setup()    # 进行数据集分割
folder_datamoduleB.setup()    # 进行数据集分割
folder_datamoduleC.setup()    # 进行数据集分割

train_loader = folder_datamoduleA.train_dataloader()
val_loader   = folder_datamoduleB.val_dataloader()
test_loader  = folder_datamoduleC.test_dataloader()



model = Fastflow()   # official: mcait, other extractors tested: resnet18, wide_resnet50_2. Could use others...
engine = Engine(task=TaskType.CLASSIFICATION,
                image_metrics=["F1Score","AUROC"], pixel_metrics=["F1Score","AUROC"],
                max_epochs=200,                            #! 希望赋值给Lightning Trainer的参数必须全部放在已标明参数的最后面
                )


engine.fit(datamodule=folder_datamoduleA, model=model)