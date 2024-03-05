import cv2
import numpy as np
import torchvision
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from build_model import ConvNeXt

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
