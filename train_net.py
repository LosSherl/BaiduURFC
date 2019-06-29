import torch
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import DataLoader
import logging
import time
import datetime

from Utils.logger import setupLogger
from Utils.checkpoint import Checkpointer
from modeling.MMmodel import MultiModalNet
from dataset.dataset_builder import MMDataset

def do_train(name, model, device, trndata_loader, valdata_loader, optimizer, criterion, scheduler, nepochs, checkpoint_period, checkpointer):
    logger = logging.getLogger(name=name)
    logger.info("Start training")
    
    total_step = len(trndata_loader)
    start_training_time = time.time()
    for epoch in range(nepochs):
        model.train()
        for iteration, (imgs, visits, labels) in enumerate(trndata_loader):
            imgs = imgs.to(device)
            visits = visits.to(device)
            # idx_labels = labels.clone()
            labels = torch.from_numpy(np.array(labels)).long().to(device)

            output = model(imgs, visits)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 50 == 0:
                logger.info(
                ", ".join(
                        [
                            "Epoch: [{epoch}/{num_epochs}]",
                            "Step: [{iter}/{total_step}",
                            "Loss: {loss:.4f}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        epoch = epoch + 1, num_epochs = nepochs,
                        iter = iteration + 1, total_step = total_step,
                        loss = loss.item(),
                        lr = optimizer.param_groups[0]["lr"],
                        memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        time_spent = time.time() - start_training_time
        logger.info("Epoch:[{}/{}], Time spent {}, Time per epoch {:.4f} s".format(
            epoch + 1, nepochs, str(datetime.timedelta(seconds=time_spent)), time_spent / (epoch + 1)))
        if (epoch + 1) % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(epoch + 1))

        # val
        model.eval()
        with torch.no_grad():
            val_loss = 0
            total = 0
            correct = 0
            for _, (imgs, visits, labels) in enumerate(valdata_loader):
                imgs = imgs.to(device)
                visits = visits.to(device)
                # idx_labels = labels.clone()
                labels = torch.from_numpy(np.array(labels)).long().to(device)
                
                output = model(imgs, visits)
                val_loss += criterion(output, labels)
                correct += accuracy_score(labels.cpu().data.numpy(),np.argmax(output.cpu().data.numpy(), axis=1),normalize=False)
                total += labels.size(0) 
        logger.info("Epoch:[{}/{}], validation loss: {}, Validation acc@1: {}%".format(
            epoch + 1, nepochs, val_loss, 100 * correct / total))   

        scheduler.step(val_loss)

    checkpointer.save("model_final")
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (nepochs)
        )
    )

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

    args = parser.parse_args()

    logger = setupLogger(args.name, args.output_dir, filename=args.name + "_log.txt")
    logger.info(args)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalNet("se_resnext101_32x4d", "dpn26", 0.5, num_classes=9, pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)
    
    all_files = pd.read_csv("train.csv")
    test_files = pd.read_csv("test.csv")
    train_datalist, val_datalist = train_test_split(all_files, test_size=0.1, random_state=2050)

    train_img = os.path.join(args.root_path, "train")
    test_img = os.path.join(args.root_path, "test")
    train_visit = os.path.join(args.root_path, "npy", "train_visit")
    test_visit = os.path.join(args.root_path, "npy", "test_visit")
    
    trndatasets = MMDataset(train_datalist, train_img, train_visit, mode="train")
    trndata_loader = DataLoader(trndatasets, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)
    
    valdatasets = MMDataset(val_datalist, train_img, train_visit, augment=False, mode="train")
    valdata_loader = DataLoader(valdatasets, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    testdatasets = MMDataset(test_files, test_img, test_visit, augment=False, mode="test")
    test_loader = DataLoader(testdatasets, 1, shuffle=False, pin_memory=True, num_workers=1)

    checkpointer = Checkpointer(
        model, optimizer, criterion, scheduler, args.output_dir,
    )

    do_train(args.name, model, device, trndata_loader, valdata_loader, optimizer, criterion, scheduler, args.nepochs, args.checkpoint_period, checkpointer)

if __name__ == "__main__":
    main()