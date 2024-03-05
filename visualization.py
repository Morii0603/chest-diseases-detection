import cv2
import numpy as np
import torchvision
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

def visualization(img_path):
# 加载模型
    model = ConvNeXt()
    model.load_state_dict(torch.load("best.pth"))
# 读入图片
    img = cv2.imread(img_path,1)
    img = cv2.resize(img,(224,224))
    img_float_np = np.float32(img) / 255
# 转换为tensor
    transform = torchvision.transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0)
# 选定目标层
    target_layer = [model.model.features[-1][-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(
        img_float_np, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_HSV
    )
    return cam_image
# 图片预处理
def preprocess_image(image):
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch
