# SENet.pytorch

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Jie Hu, Li Shen and Gang Sun, who are the winners of ILSVRC 2017 classification competition.

Now SE-ResNet (18, 34, 50, 101, 152/20, 32) and SE-Inception-v3 are implemented.

* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset.

* `python imagenet.py` and `python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} imagenet.py` run SE-ResNet50 with ImageNet(2012) dataset,
    + You need to prepare dataset by yourself in `~/.torch/data` or set an enviroment variable `IMAGENET_ROOT=${PATH_TO_YOUR_IMAGENET}`
    + First download files and then follow the [instruction](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).
    + The number of workers and some hyper parameters are fixed so check and change them if you need.
    + This script uses all GPUs available. To specify GPUs, use `CUDA_VISIBLE_DEVICES` variable. (e.g. `CUDA_VISIBLE_DEVICES=1,2` to use GPU 1 and 2)

For SE-Inception-v3, the input size is required to be 299x299 [as the original Inception](https://github.com/tensorflow/models/tree/master/inception).

## Pre-requirements

The codebase is tested on the following setting.

* Python>=3.8
* PyTorch>=1.6.0
* torchvision>=0.7

### For training

To run `cifar.py` or `imagenet.py`, you need

* `pip install git+https://github.com/moskomule/homura@v2020.07`

## hub

You can use some SE-ResNet (`se_resnet{20, 56, 50, 101}`) via `torch.hub`.

```python
import torch.hub
hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet20',
    num_classes=10)
```

Also, a pretrained SE-ResNet50 model is available.

```python
import torch.hub
hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)
 ```

## Results

### SE-ResNet20/Cifar10

```
python cifar.py [--baseline]
```

Note that the CIFAR-10 dataset expected to be under `~/.torch/data`.

|                  | ResNet20       | SE-ResNet20 (reduction 4 or 8)    |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 93%            |

### SE-ResNet50/ImageNet

```
python [-m torch.distributed.launch --nproc_per_node=${NUM_GPUS}] imagenet.py
```

The option [-m ...] is for distributed training. Note that the Imagenet dataset is expected to be under `~/.torch/data` or specified as `IMAGENET_ROOT=${PATH_TO_IMAGENET}`.

*The initial learning rate and mini-batch size are different from the original version because of my computational resource* .

|                  | ResNet         | SE-ResNet      |
|:-------------    | :------------- | :------------- |
|max. test accuracy(top1)|  76.15 %(*)             | 77.06% (**)          |


+ (*): [ResNet-50 in torchvision](https://pytorch.org/docs/stable/torchvision/models.html)

+ (**): When using `imagenet.py` with the `--distributed` setting on 8 GPUs. The weight is [available](https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl).

```python
# !wget https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl

senet = se_resnet50(num_classes=1000)
senet.load_state_dict(torch.load("seresnet50-60a8950a85b2b.pkl"))
```

## Contribution

I cannot maintain this repository actively, but any contributions are welcome. Feel free to send PRs and issues.

## References

[paper](https://arxiv.org/pdf/1709.01507.pdf)

[authors' Caffe implementation](https://github.com/hujie-frank/SENet)
