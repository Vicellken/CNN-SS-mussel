# Guideline

Code for the study:
```
Gu et al., 2023, A comparative study on CNN-based semantic segmentation of intertidal mussel beds. Ecological Informatics
```

If you find this work useful for your research, please cite as:
```

```

## 1. Setup environment

Install Anaconda or miniconda to your device. See more information at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

```
# choose your own virtual environment name
conda create --name paddlex_env python=3.8
'''
# if you encountered package errors please consider use
conda create --name paddlex_env python=3.7
'''

conda activate paddlex_env
```

Config PaddlePaddle environment. Visit [this link](https://www.paddlepaddle.org.cn/en) for more instructions.

```
# if your device has dedicated GPU
# check the CUDA version on your device
nvidia-smi

# select the suitable CUDA version
# be aware CPU version is also available
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

Config PaddleX. The following steps are based on API development mode on UNIX system, for different platform please see [more instructions](https://github.com/PaddlePaddle/PaddleX).

```
pip install paddlex==2.1.0 -i https://mirror.baidu.com/pypi/simple
```

```
# for Linux or MacOS
pip install cython
pip install pycocotools

# for Windows, Microsoft Visual C++ 14.0 or greater is required.
# Get it with "Microsoft C++ Build Tools": 
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

<br />

## 2. Prepare training dataset

Annotating images used to be iterative and time consuming. This study used [EISeg (Efficient Interactive Segmentation)](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/EISeg/README_EN.md), open sourced by PaddlePaddle (see Materials and methods 2.1).

Install the EISeg tool:

```
pip install eiseg
```

For instructions on EISeg interactive annotation process, see [this link](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/EISeg/docs/image_en.md).

A demo of annotated image is available in the directory `./demo dataset`. However, the dataset needs to be converted to PaddleX compatible format. For example:

```
Dataset/ # Dataset directory
|--JPEGImages/ # Raw images
|  |--1.jpg
|  |--2.jpg
|  |--...
|  |--...
|
|--Annotations/ # Annotated images
|  |--1.png
|  |--2.png
|  |--...
|  |--...

|--labels.txt
background
mussel


|--train_list.txt
JPEGImages/1.jpg Annotations/1.png
JPEGImages/2.jpg Annotations/2.png
... ...


|--val_list.txt
JPEGImages/1.jpg Annotations/1.png
JPEGImages/2.jpg Annotations/2.png
... ...
```

Convert the annotated dataset to PaddleX compatible format. The function is available at the develop branch of the original repository, please see [this link](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/EISeg/tool/eiseg2paddlex.py).

```
cd utils
python eiseg2paddlex.py
```

If you have separated image annotation jobs or want to extend the existing labelling data

```
cd utils
python merge_coco.py
```

<br />

## 3. Image augmentation

This step is included in the experiment models (see Python code in the directory `./experiment models`).

For detailed information on model API, please see [this link](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/cv/transforms/operators.py).

<br />

## 4. Model training and evaluation

Each model has two sets of scripts where the Bash script (.sh file) configs the parallel GPUs environment; Python script (.py file) includes the:

- Image augmentation
- Dataset path
- Model configurations
- Training hyper-parameters

More details about defining the model segmenters are available at [this link](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/cv/models/segmenter.py).

If your training environment cannot access the Internet for system integrity reasons, please download the pre-trained weights beforehand and direct the path to pre-trained weights in the Python script. For more information please see [this link](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/utils/checkpoint.py).

```
# download the Cityscapes pre-trained weights
cd utils
python get_pretain_weights.py
```

<br />

#### Reach me by email for more discussion. Happy to chat ;)
