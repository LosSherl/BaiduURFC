import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import shutil
import os
from tqdm import tqdm
import torchvision.transforms as tr
import copy as cp


def calc_black_ratio(img_path):
    Img = cv2.imread(img_path)
    Gray = tr.Compose([
                       tr.ToPILImage(),
                       tr.Grayscale(),
                       tr.ToTensor()
                       ])

    Img_Gray = Gray(Img)
    ratio = float(torch.eq(Img_Gray, 0.0).sum())/10000.0

    return ratio

def calc_white_ratio(img_path):
    Img = cv2.imread(img_path)
    Gray = tr.Compose([
                       tr.ToPILImage(),
                       tr.Grayscale(),
                       tr.ToTensor()
                       ])

    Img_Gray = Gray(Img)
    ratio = float(torch.eq(Img_Gray, 255.0).sum())/10000.0

    return ratio

if __name__ == '__main__':
    total = 0
    bad_cnt = 0
    path = "/dataset/2019_big_data_competition_final"
    bad_dir = os.path.join(path, "bad_samples")
    f = open("/code/cl/BaiduURFC/train.csv", "w")
    f.write("Id,Target\n")
    path = os.path.join(path, "train")
    if not os.path.exists(bad_dir):
        os.makedirs(bad_dir)
    for d in tqdm(os.listdir(path)):
        cur_dir = os.path.join(path, d)
        for img in os.listdir(cur_dir):
            total += 1
            abs_path = os.path.join(path, d, img)
            ratio = calc_black_ratio(abs_path)
            if ratio > 0.25:
                shutil.copyfile(abs_path, os.path.join(bad_dir, img))
                bad_cnt += 1
                continue
            ratio = calc_white_ratio(abs_path)
            if ratio > 0.9:
                bad_cnt += 1
                shutil.copyfile(abs_path, os.path.join(bad_dir, img))
                continue
            f.write(d + "/" + img.split(".")[0] + "," + str(int(d[-1]) - 1) + "\n")
    print("total: ", str(total))
    print("bad: ", str(bad_cnt))













