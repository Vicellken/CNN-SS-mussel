import os.path as osp
from paddlex.utils.checkpoint import cityscapes_weights
from paddlex.utils.download import download_and_decompress

save_dir = 'pretrained_weights'  # Path to save pre-trained weights

weights_lists = [cityscapes_weights, ]
for weights in weights_lists:
    for key, value in weights.items():
        new_save_dir = osp.join(save_dir, key)
        download_and_decompress(value, path=new_save_dir)
