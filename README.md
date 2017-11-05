# SENet.pytorch

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Jie Hu, Li Shen and Gang Sun, who are the winners of ILSVRC 2017 classification competition.


Now SE-ResNet (18, 34, 50, 101, 152/20, 32) and SE-Inception-v3 are implemented.

* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset.

* `python imagenet.py IMAGENET_ROOT` runs SE-ResNet50 with ImageNet(2012) dataset.
    + You need to prepare dataset by yourself
    + First download files and then follow the [instruction](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

For SE-Inception-v3, the input size is required to be 299x299 [as original Inception](https://github.com/tensorflow/models/tree/master/inception).

## result

### SE-ResNet20/Cifar10

|                  | ResNet20       | SE-ResNet20    |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | -              |

### SE-ResNet50/ImageNet

|                  | ResNet         | SE-ResNet      |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  -             | -              |

## references

[paper](https://arxiv.org/pdf/1709.01507.pdf)

[authors' Caffe implementation](https://github.com/hujie-frank/SENet)
