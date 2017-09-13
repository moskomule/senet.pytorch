# SENet.pytorch

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Jie Hu, Li Shen and Gang Sun, who are the winners of ILSVRC 2017 classification competition.


Now SE-ResNet (18, 34, 50, 101, 152) and SE-Inception-v3 are implemented.

`python exec.py` runs SE-ResNet50 with Cifar10 dataset.

For SE-Inception-v3, the input size is required to be 299x299 [as original Inception](https://github.com/tensorflow/models/tree/master/inception).

## references

[paper](https://arxiv.org/pdf/1709.01507.pdf)
[authors' Caffe implementation](https://github.com/hujie-frank/SENet)
