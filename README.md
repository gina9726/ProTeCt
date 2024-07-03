# ProTeCt [[project page]](http://www.svcl.ucsd.edu/projects/protect)
This repository contains the source code accompanying our CVPR 2024 paper.  

[<b>ProTeCt: Prompt Tuning for Taxonomic Open Set Classification</b>](https://arxiv.org/pdf/2306.02240)  
[Tz-Ying Wu](http://www.svcl.ucsd.edu/people/gina), [Chih-Hui Ho](http://www.svcl.ucsd.edu/people/johnho), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno)

# Dependencies
- Python (3.8.16)
- PyTorch (1.12.1)
- torchvision (0.13.1)
- NumPy (1.23.5)
- Pillow (9.4.0)
- PyYaml (6.0)
- tensorboardX (2.6)
We also provide the pre-built conda environment for easier setup. Simply run
```
$ conda env create -f environment.yml
$ conda activate ProTeCt
```

# Data preparation
- CIFAR100 
[[Raw images]](https://www.cs.toronto.edu/~kriz/cifar.html)
- SUN
[[Raw_images]](https://vision.princeton.edu/projects/2010/SUN/)
- ImageNet
[[Raw images]](http://image-net.org/download-images)

These datasets can be downloaded from the above links.
Please put the raw images in the folder `prepro/raw/`. We have preprocessed the data lists (`gt_{split}.txt`), hierarchy (`tree.npy`), and treecuts for evaluating MTA (`treecuts_*.pkl`) in `prepro/data/` using the splits in `prepro/splits/`. Only the former is required for running the code, and the latter is just for reference. Note that `prepro/prepro.py` can be used if you want to build the data lists and the tree hierarchies of other datasets. 

# Training
To train the model with different datasets and model configurations (e.g. CoOp/MaPLe/CoOp+ProTeCt/MaPLe+ProTeCt), you need to indicate the config file. We provide the config templates in the folder `configs`. For example,
```
$ python train.py --config configs/{dataset}/few_shot/16_shot/CoOp+ProTeCt.yml --trial 1
```
where {dataset}=cifar100/sun/imagenet.
Note that training will automatically create log files (including txt and tfevents) and model checkpoints, and the default folder is under `runs/`.

# Evaluation
To evaluate a model, you need to indicate the folder of the experiment. For example,
```
$ python test.py --folder runs/{dataset}/coop/ViT-B_16/few_shot/16-shot/CoOp/trial_1 --bz {batch_size}
```
where {dataset}=cifar100/sun/imagenet, and you can set the batch size for testing with {batch_size} based on your GPU memory.

For testing across datasets for the imagenet transfer experiments, you need to set the {eval_dataset}
```
$ python test.py --folder runs/{dataset}/coop/ViT-B_16/few_shot/16-shot/CoOp/trial_1 --bz {batch_size} --eval_dataset {eval_dataset}
```
where {eval_dataset}=imagenet-a/imagenet-r/imagenetV2/imagenet-sketch.



# Citation
If you find this repository useful, please consider cite our paper.
```
@InProceedings{Wu_2024_CVPR,
author = {Wu, Tz-Ying and Ho, Chih-Hui and Vasconcelos, Nuno},
title = {ProTeCt: Prompt Tuning for Taxonomic Open Set Classification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2024}
}
```
