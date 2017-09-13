# SENet.pytorch

An implementation of SENet, proposed in **Squeeze-and-Excitation Networks** by Hu, J., Shen, L., & Sun, G. (n.d.), who are the winners of ILSVRC 2017 classification competition.

Now SE-ResNet (18, 34, 50, 101, 152) and SE-Inception-v3 are implemented.

`python exec.py` runs SE-ResNet50 with Cifar10 dataset.

For SE-Inception-v3, the input size is required to be 299x299 [as original Inception](https://github.com/tensorflow/models/tree/master/inception).