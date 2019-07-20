import torch
import os
from tqdm import tqdm
import numpy as np

def test_submit(model, test_loader, device, output_dir="."):
    model.to(device)
    model.eval()
    res = dict()
    f = open(os.path.join(output_dir, "submit.txt"), "w")
    for i, (imgs, visits, filepath) in tqdm(enumerate(test_loader)):
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            imgs = imgs.to(device)
            visits = visits.to(device)
            y_pred = model(imgs, visits)
            labels = np.argmax(y_pred.cpu().data.numpy(), axis=1)
            for j in range(imgs.size(0)): 
                res[filepath[j]] = filepath[j] + "\t00" + str(labels[j] + 1) + "\n"
    for i in range(100000):
        key = str(i).zfill(6)
        f.write(res[key])

def ensemble_submit(models, test_loader, device, output_dir="."):
    for model in models:
        model.to(device)
        model.eval()
    res = dict()
    f = open(os.path.join(output_dir, "submit.txt"), "w")
    for i, (imgs, visits, filepath) in tqdm(enumerate(test_loader)):
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            imgs = imgs.to(device)
            visits = visits.to(device)
            output = []
            for model in models:
                t = torch.nn.functional.normalize(model(imgs, visits))
                output.append(t)
            output = sum(output)
            labels = np.argmax(output.cpu().data.numpy(), axis=1)
            for j in range(imgs.size(0)): 
                res[filepath[j]] = filepath[j] + "\t00" + str(labels[j] + 1) + "\n"
    for i in range(100000):
        key = str(i).zfill(6)
        f.write(res[key])