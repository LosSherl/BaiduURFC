import torch
import argparse

from utils.logger import setupLogger

def main():
    parser = argparse.ArgumentParser(description="Baidu URFC")
    parser.add_argument(
        "-p", 
        dest = "data_path",
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
        default = 10,
        type = int
    )
    parser.add_argument(
        "-lr",
        dest = "lr",
        default = 0.01,
        type = float
    )

    args = parser.parse_args()

    logger = setupLogger("UFRC", args.save_dir)
    logger.info(args)

    

if __name__ == "__main__":
    main()