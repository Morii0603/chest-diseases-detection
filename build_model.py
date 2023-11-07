# encoding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ConvNeXt(nn.Module):
    def __init__(self, out_features=14, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            self.model = torchvision.models.convnext_tiny()
        self.model.classifier[2] = nn.Linear(768, out_features=out_features)
        self.model.classifier.add_module("sigmoid", nn.Sigmoid())
    def forward(self, x):
        x = self.model(x)
        return x

def ConvNeXt_T(out_features=14, pretrained=True):
    if pretrained:
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
    else:
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(768, out_features=out_features)
    model.classifier.add_module("sigmoid", nn.Sigmoid())
    return model

def ResNet50(out_features=14, pretrained=True):
    if pretrained:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.resnet50()
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, out_features),
        nn.Sigmoid()
    )
    return model

def Swin_T(out_features=14, pretrained=True):
    if pretrained:
        model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
    else:
        model = torchvision.models.swin_t()
    model.head = nn.Sequential(
        nn.Linear(768, out_features=out_features),
        nn.Sigmoid()
    )
    return model


