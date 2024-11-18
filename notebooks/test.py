from pathlib import Path
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import torch

from anomalib import TaskType
from anomalib.data import MVTec, PredictDataset
from anomalib.data.image.folder import Folder, FolderDataset

from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.utils.post_processing import superimpose_anomaly_map


torch.set_float32_matmul_precision('medium')

dataset_root = r"F:\\Projects\\anomalib\\datasets\\3-5"


folder_datamodule = Folder(
    name="3-5",
    root=dataset_root,
    normal_dir="normal",
    abnormal_dir="abnormal",
    task=TaskType.CLASSIFICATION,
    image_size=(256, 256),
    num_workers=2,
    val_split_ratio=0.3,
    normal_split_ratio = 0.2,
    train_batch_size = 2,
    eval_batch_size = 2,
    seed = 1,
)
folder_datamodule.setup()



i, data = next(enumerate(folder_datamodule.train_dataloader()))
print("train",data.keys(), data["image"].shape)

i, data = next(enumerate(folder_datamodule.val_dataloader()))
print("val",data.keys(), data["image"].shape)

i, data = next(enumerate(folder_datamodule.test_dataloader()))
print("test",data.keys(), data["image"].shape)



# Import the model and engine
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Create the model and engine
model = Patchcore()
engine = Engine(task="classification")

# Train a Patchcore model on the given datamodule
engine.train(datamodule=folder_datamodule, model=model)




# from anomalib.models import Padim
# model = Padim(
#     backbone="resnet18",
#     layers=["layer1", "layer2", "layer3"],
# )



# from anomalib.engine import Engine
# from anomalib.utils.normalization import NormalizationMethod

# engine = Engine(
#     normalization=NormalizationMethod.MIN_MAX,
#     threshold="F1AdaptiveThreshold",
#     task=TaskType.CLASSIFICATION,
#     image_metrics=["AUROC"],
#     accelerator="auto",
#     check_val_every_n_epoch=1,
#     devices=1,
#     max_epochs=3,
#     num_sanity_val_steps=0,
#     val_check_interval=1.0,
# )



# engine.fit(datamodule=datamodule, model=model)


# engine.test(datamodule=datamodule, model=model)
