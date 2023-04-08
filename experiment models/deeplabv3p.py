import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # set GPU id, if you have single GPU, set '0'
import paddle
paddle.set_device("gpu")
import paddlex as pdx
from paddlex import transforms as T

'''
Image augumentation
for more details, please refer to:
https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/cv/transforms/operators.py
'''
train_transforms = T.Compose([
    T.RandomDistort(),
    T.RandomVerticalFlip(),
    T.RandomHorizontalFlip(),
    T.Normalize(),
    T.Resize(target_size=1024)
])

eval_transforms = T.Compose([
    T.Normalize(),
    T.Resize(target_size=1024)
])

'''
Dataset path
'''
train_dataset = pdx.datasets.SegDataset(
    data_dir='dataset',
    file_list='dataset/train_list.txt',
    label_list='dataset/labels.txt',
    transforms=train_transforms
)

eval_dataset = pdx.datasets.SegDataset(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms
)

'''
Model configurations
for more details, please refer to:
https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/cv/models/segmenter.py
'''
num_classes = len(train_dataset.labels)
model = pdx.seg.DeepLabV3P(
    num_classes=num_classes,
    backbone='ResNet50_vd',
    use_mixed_loss=False,
    output_stride=8,
    backbone_indices=(0, 3),
    aspp_ratios=(1, 12, 24, 36),
    aspp_out_channels=256,
    align_corners=False)

'''
Training hyper-parameters
for more details, please refer to:
https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/cv/models/segmenter.py
'''
model.train(
    num_epochs=1000,
    train_dataset=train_dataset,
    train_batch_size=8, # batch size should be set according to the GPU memory
    eval_dataset=eval_dataset,
    optimizer=None, # by default, the optimizer is Momentum
    save_interval_epochs=int(2), # save model every 2 epochs
    log_interval_steps=int(5), # log training process every 5 steps
    save_dir='output_deeplabv3p',
    pretrain_weights="CITYSCAPES", # without pretraining, set `pretrain_weights = None`
    learning_rate=0.001,
    # if you want to stop training when a metric has stopped improving, set `early_stop` to True
    early_stop=True,
    early_stop_patience=int(10),
    use_vdl=True  # if you want to use VisualDL to log training process, set `use_vdl` to True
)
