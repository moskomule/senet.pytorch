# SENet.pytorch

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Jie Hu, Li Shen and Gang Sun, who are the winners of ILSVRC 2017 classification competition.


Now SE-ResNet (18, 34, 50, 101, 152/20, 32) and SE-Inception-v3 are implemented.

* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset.

* `python imagenet.py IMAGENET_ROOT` runs SE-ResNet50 with ImageNet(2012) dataset.
    + You need to prepare dataset by yourself
    + First download files and then follow the [instruction](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).
    + The number of workers and some hyper parameters are fixed so check and change them if you need.
    + This script uses all GPUs available. To specify GPUs, use `CUDA_VISIBLE_DEVICES` variable. (e.g. `CUDA_VISIBLE_DEVICES=1,2` to use GPU 1 and 2)

For SE-Inception-v3, the input size is required to be 299x299 [as the original Inception](https://github.com/tensorflow/models/tree/master/inception).

## Pre-requirements

* Python>=3.6
* PyTorch>=0.4
* torchvision
* tqdm

## Result

### SE-ResNet20/Cifar10

```
python cifar.py [--baseline]
```

|                  | ResNet20       | SE-ResNet20 (reduction 4 or 8)    |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 93%            |

### SE-ResNet50/ImageNet

*The initial learning rate and mini-batch size are different from the original version because of my computational resource* (0.6 to 0.1 and 1024 to 128 respectively).

|                  | ResNet         | SE-ResNet      |
|:-------------    | :------------- | :------------- |
|max. test accuracy(top1)|  79.26 %(*)             | 71.66 %(**)          |


+ (*): He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition.

+ (**): I share [this weight (training after 100 epochs)](https://drive.google.com/file/d/1WhBKRKIRtd-Fsrj3hx_WNycdsZSbC9Ep).

```python
senet = se_resnet50(num_classes=1000)
senet.load_state_dict(torch.load("weight.pkl"))
```

## References

[paper](https://arxiv.org/pdf/1709.01507.pdf)

[authors' Caffe implementation](https://github.com/hujie-frank/SENet)
