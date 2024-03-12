# encoding: utf-8
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from tqdm import tqdm
from build_model import ConvNeXt
from utils import compute_AUCs, CosineWarmupScheduler

epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip()])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
train_dataset = ChestXrayDataSet(
    data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=train_transform
)
val_dataset=ChestXrayDataSet(
    data_dir=DATA_DIR, image_list_file=VAL_IMAGE_LIST, transform=test_transform
)
test_dataset = ChestXrayDataSet(
    data_dir=DATA_DIR, image_list_file=TEST_IMAGE_LIST, transform=test_transform
)

model = ConvNeXt()
model = model.to(device)
train_loader = DataLoader(train_dataset,batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
criterion = nn.BCELoss()


optimizer = torch.optim.AdamW(lr=0.0001, params=model.parameters())
scheduler = CosineWarmupScheduler(optimizer, total_steps=50, warmup_steps=4, min_lr=0.00001)

total_train_loss = []
total_valid_loss = []




for epoch in range(epochs):
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    itr = 1
    train_loss = []
    pbar = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        loss_value = loss.item()
        train_loss.append(loss_value)
        optimizer.step()
        pbar.set_description(f"Epoch: {epoch+1}, Batch: {itr}, Loss: {loss_value}")
        itr += 1
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)

    scheduler.step()

    with torch.no_grad():
        model.eval()
        pbar = tqdm(test_loader)
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            gt = torch.cat((gt, targets), 0)
            outputs = model(inputs)
            pred = torch.cat((pred, outputs), 0)
        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
    print(f"Epoch Completed: {epoch+1}/{epochs}, "
        f"Train Loss: {epoch_train_loss}, Valid_AUC: {AUROC_avg}")


