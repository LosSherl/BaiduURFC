import torch
import logging
import time
import datetime
from sklearn.metrics import accuracy_score
import numpy as np


def val(model, val_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for _, (imgs, visits, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            visits = visits.to(device)
            labels = torch.from_numpy(np.array(labels)).long().to(device)
            
            output = model(imgs, visits)
            correct += accuracy_score(labels.cpu().data.numpy(),np.argmax(output.cpu().data.numpy(), axis=1), normalize=False)
            total += labels.size(0) 
    acc = 100 * correct / total
    print("Validation acc@1: {}%".format(acc))

def ensemble_val(models, val_loader, device):
    for model in models:
        model = model.to(device)
        model.eval()
    with torch.no_grad():
        for i, (imgs, visits, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            visits = visits.to(device)
            labels = torch.from_numpy(np.array(labels)).long().to(device)
            output = []
            for model in models:
                t = torch.nn.functional.normalize(model(imgs, visits))
                output.append(t)
            output = sum(output)
            correct += accuracy_score(labels.cpu().data.numpy(), np.argmax(output.cpu().data.numpy(), axis=1), normalize=False)
            total += labels.size(0)
        acc = 100 * correct / total
        print("Validation acc@1: {}%".format(acc))