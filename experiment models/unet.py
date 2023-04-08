import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import paddle
paddle.set_device("gpu")
import paddlex as pdx
from paddlex import transforms as T

'''
Image augumentation
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
'''
num_classes = len(train_dataset.labels)
model = pdx.seg.UNet(
    num_classes=num_classes,
    use_mixed_loss=False,
    align_corners=False)

'''
Training hyper-parameters
'''
model.train(
    num_epochs=1000, 
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    optimizer=None,
    save_interval_epochs=int(2),
    log_interval_steps=int(5),
    save_dir='output_unet',
    pretrain_weights='CITYSCAPES',
    learning_rate=0.001,
    early_stop=True,
    early_stop_patience=int(10),
    use_vdl=True
)
