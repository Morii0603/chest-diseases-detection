# encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from tqdm import tqdm
from build_model import *
from utils import compute_AUCs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "model.pth.tar"
N_CLASSES = 14
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
DATA_DIR = "./ChestX-ray14/images"
TEST_IMAGE_LIST = "./ChestX-ray14/labels/test_list.txt"
TRAIN_IMAGE_LIST = "ChestX-ray14/labels/train_list.txt"
VAL_IMAGE_LIST = "ChestX-ray14/labels/val_list.txt"
BATCH_SIZE = 64
NUM_EPOCHS = 10


model = ConvNeXt()
model = model.to(device)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


test_dataset = ChestXrayDataSet(
    data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model.load_state_dict(torch.load("checkpoints/best"))

gt = torch.FloatTensor()
gt = gt.to(device)
pred = torch.FloatTensor()
pred = pred.to(device)

with torch.no_grad():
    valid_loss = []
    model.eval()
    pbar = tqdm(test_loader)
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        gt = torch.cat((gt, targets), 0)
        outputs = model(inputs)
        pred = torch.cat((pred, outputs), 0)


AUROCs = compute_AUCs(gt, pred)
AUROC_avg = np.array(AUROCs).mean()

print(np.array(AUROCs))
print(AUROC_avg)
