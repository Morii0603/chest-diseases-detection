from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt
from build_model import ConvNeXt


img_path = "00005794_000.png"
model = ConvNeXt()
model.load_state_dict(torch.load("Epoch_4"))

target_layer = [model.model.features[-1][-1]]

img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))

img = np.array(img)
img_float_np = np.float32(img) / 255

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)


input_tensor = transform(img)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

input_tensor = input_tensor.unsqueeze(0)

cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

targets = None
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(
    img_float_np, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_HSV
)
cv2.imwrite("localization/test.jpeg", cam_image)
