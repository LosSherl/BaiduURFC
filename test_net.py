import torch
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from torch.utils.data import DataLoader
import logging
import time
import datetime

from tools.logger import setupLogger
from tools.checkpoint import Checkpointer
from modeling.MMmodel import MultiModalNet
from dataset.dataset_builder import MMDataset
from engine.test import test_submit
from engine.val import val

def main():
    parser = argparse.ArgumentParser(description="Baidu URFC")
    parser.add_argument(
        "-p", 
        dest = "root_path",
        help = "path to data"
    )
    parser.add_argument(
        "-o",
        dest = "output_dir",
        help = "output dir",
        default = "ouputs",
    )
    parser.add_argument(
        "-ckt",
        dest = "ckt"
    )
    parser.add_argument(
        "-name",
        dest = "name",
        default = "URFC"
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.ckt, map_location=torch.device("cpu"))
    model = MultiModalNet("se_resnext101_32x4d", "dpn26", 0.5, num_classes=9, pretrained=True)
    model.load_state_dict(checkpoint["model"])
    print("model loaded from ", args.ckt)

    val_files = pd.read_csv("train.csv")
    val_img = os.path.join(args.root_path, "train")
    val_visit = os.path.join(args.root_path, "npy", "train_visit")
    
    kf = KFold(n_splits=10, random_state=2050)
    splits = []
    for train_list, test_list in kf.split(val_files):
        splits.append((train_list, test_list))
    val_files.drop(splits[0][0])
    valdatasets = MMDataset(val_files, val_img, val_visit, augment=False, mode="val")
    val_loader = DataLoader(valdatasets, 128, shuffle=False, pin_memory=True, num_workers=1)

    test_files = pd.read_csv("test.csv")
    test_img = os.path.join(args.root_path, "test")
    test_visit = os.path.join(args.root_path, "npy", "test_visit")
    testdatasets = MMDataset(test_files, test_img, test_visit, augment=False, mode="test")
    test_loader = DataLoader(testdatasets, 1, shuffle=False, pin_memory=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val(model, val_loader, device)
    test_submit(model, test_loader, device)

if __name__ == "__main__":
    main()