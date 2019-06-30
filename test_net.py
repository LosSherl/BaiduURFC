import torch
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
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
    test_files = pd.read_csv("test.csv")
    test_img = os.path.join(args.root_path, "test")
    test_visit = os.path.join(args.root_path, "npy", "test_visit")
    testdatasets = MMDataset(test_files, test_img, test_visit, augment=False, mode="test")
    test_loader = DataLoader(testdatasets, 1, shuffle=False, pin_memory=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_submit(model, test_loader, device)

if __name__ == "__main__":
    main()