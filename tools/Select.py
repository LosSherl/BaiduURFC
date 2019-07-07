import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import shutil
import os
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
    path = "/dataset/2019_big_data_competition_final/train"
    f = open("../train.csv", "w")
    f.write("Id,Target")
    bad_dir = os.path.join(path, "bad_samples")
    if not os.path.exists(bad_dir):
        os.makedirs(bad_dir)
    for d in os.listdir(path):
        cur_dir = os.path.join(path, d)
        for img in os.listdir(cur_dir):
            abs_path = os.path.abspath(img)
            ratio = calc_black_ratio(abs_path)
            if ratio > 0.25:
                shutil.copyfile(abs_path, os.path.join(bad_dir, img))
                continue
            ratio = calc_white_ratio(abs_path)
            if ratio > 0.9:
                shutil.copyfile(abs_path, os.path.join(bad_dir, img))
                continue
            f.write(d + "/" + img + "," + d[-1])













