from torch import nn
from torchvision.models import googlenet
from torchsummary import summary
import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from senet.se_module import SELayer



class SEGoogLeNet(nn.Module):
    def __init__(self, num_classes, pretrained, aux_logits=True, transform_input=False):
        super(SEGoogLeNet, self).__init__()
        #Pretrain and customize classifier test
        model = googlenet(pretrained = pretrained, aux_logits = aux_logits, transform_input = transform_input)
        in_c = model.fc.in_features
        model.fc = nn.Linear(in_c, num_classes)
        if aux_logits:
            model.aux1 = InceptionAux(512, num_classes)
            model.aux2 = InceptionAux(528, num_classes)

		
        #Add SELayers to each inception module
        model.inception3a.add_module("SELayer", SELayer(32, 16))
        model.inception3b.add_module("SELayer", SELayer(64, 16))
        model.inception4a.add_module("SELayer", SELayer(64, 16))
        model.inception4b.add_module("SELayer", SELayer(64, 16))
        model.inception4c.add_module("SELayer", SELayer(64, 16))
        model.inception4d.add_module("SELayer", SELayer(64, 16))
        model.inception4e.add_module("SELayer", SELayer(128, 16))
        model.inception5a.add_module("SELayer", SELayer(128, 16))
        model.inception5b.add_module("SELayer", SELayer(128, 16))
        if aux_logits:
            if num_classes//16 < 1:
	            ratio = num_classes
				model.aux1.add_module("SELayer", SELayer(num_classes, ratio))
				model.aux2.add_module("SELayer", SELayer(num_classes, ratio))
	        else:
	            model.aux1.add_module("SELayer", SELayer(num_classes))
	            model.aux2.add_module("SELayer", SELayer(num_classes))
  
        self.model = model

    def forward(self, x):
        return self.model(x)




def se_googlenet(**kwargs):
    return SEGoogLeNet(**kwargs)