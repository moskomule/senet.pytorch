dependencies = ["torch", "math"]


def se_resnet20(**kwargs):
    from senet.se_resnet import se_resnet20 as _se_resnet20

    return _se_resnet20(**kwargs)


def se_resnet56(**kwargs):
    from senet.se_resnet import se_resnet56 as _se_resnet56

    return _se_resnet56(**kwargs)


def se_resnet50(**kwargs):
    from senet.se_resnet import se_resnet50 as _se_resnet50

    return _se_resnet50(**kwargs)


def se_resnet101(**kwargs):
    from senet.se_resnet import se_resnet101 as _se_resnet101

    return _se_resnet101(**kwargs)
