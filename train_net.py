import torch
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
import logging
import time
import datetime

from tools.logger import setupLogger
from tools.checkpoint import Checkpointer
from modeling.MMmodel import MultiModalNet
from dataset.dataset_builder import MMDataset
from engine.trainer import do_train
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
        "-bs",
        dest = "batch_size",
        default = 64,
        type = int
    )
    parser.add_argument(
        "-n",
        dest = "nepochs",
        default = 30,
        type = int
    )
    parser.add_argument(
        "-cp",
        dest = "checkpoint_period",
        default = 1,
        type = int
    )
    parser.add_argument(
        "-lr",
        dest = "lr",
        default = 0.01,
        type = float
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
    parser.add_argument(
        "-s",
        type = int,
        dest = "split",
        default = 0
    )

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.name) 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger = setupLogger(args.name, output_dir, filename=args.name + "_log.txt")
    logger.info(args)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalNet("se_resnext101_32x4d", "dpn26", 0.5, num_classes=9, pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)
    
    train_files = pd.read_csv("train.csv")
    test_files = pd.read_csv("test.csv")
    kf = KFold(n_splits=10, random_state=2050)
    splits = []
    for train_list, test_list in kf.split(train_files):
        splits.append((train_list, test_list))
    
    val_files = train_files.drop(splits[args.split][0])
    train_files = train_files.drop(splits[args.split][1])

    # train_datalist, val_datalist = train_test_split(all_files, test_size=0.1, random_state=2050)

    train_img = os.path.join(args.root_path, "train")
    test_img = os.path.join(args.root_path, "test")
    train_visit = os.path.join(args.root_path, "npy", "train_visit")
    test_visit = os.path.join(args.root_path, "npy", "test_visit")
    
    trndatasets = MMDataset(train_files, train_img, train_visit, mode="train")
    trndata_loader = DataLoader(trndatasets, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    
    valdatasets = MMDataset(val_files, train_img, train_visit, augment=False, mode="val")
    valdata_loader = DataLoader(valdatasets, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    testdatasets = MMDataset(test_files, test_img, test_visit, augment=False, mode="test")
    test_loader = DataLoader(testdatasets, 1, shuffle=False, pin_memory=True, num_workers=1)

    checkpointer = Checkpointer(
        model, optimizer, criterion, scheduler, output_dir,
    )

    do_train(args.name, model, device, trndata_loader, valdata_loader, optimizer, criterion, scheduler, args.nepochs, args.checkpoint_period, checkpointer)
    model = checkpointer.load(os.path.join(output_dir, "best_model.pth"))
    test_submit(model, test_loader, device, output_dir)

if __name__ == "__main__":
    main()