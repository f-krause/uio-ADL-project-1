import argparse
import torch
import numpy as np
import timm
import time
import os
from tqdm import tqdm
from torchvision import transforms
from timm import optim
from lora import LoRATransformer

from get_data import get_dataloaders_educloud, get_dataloaders_local
from train import run_full_tuning, run_lora_tuning





parser = argparse.ArgumentParser(description="Specify parameters and tasks of KNN based genre estimation")
parser.add_argument("-l", "--lora", action="store_true", help="run LoRA based training")
parser.add_argument("-s", "--save", action="store_true", help="save model in output directory")
parser.add_argument("-e", "--epochs", default=None, type=int, help="specify number of epochs")
args = parser.parse_args()


def main():
    BATCH_SIZE = 64
    INPUT_SHAPE = (224, 224, 3)

    if os.path.isdir("/projects/ec232/data/"):
        # Load data from educloud server if path exists
        train_loader, test_loader = get_dataloaders_educloud(BATCH_SIZE, INPUT_SHAPE)
    else:
        # Otherwise load local data from "imagewoof" directory
        train_loader, test_loader = get_dataloaders_local(BATCH_SIZE, INPUT_SHAPE)

    if not args.lora:
        # Exercise 1 - full fine-tuning
        run_full_tuning(train_loader, test_loader, args.save)
    else:
        # Exercise 2 - LoRA approach
        run_lora_tuning(train_loader, test_loader, args.save)


if __name__ == '__main__':
    main()
