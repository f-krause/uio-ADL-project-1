import argparse
from datetime import datetime
import os

from get_data import get_dataloaders_educloud, get_dataloaders_local
from train import run_full_tuning, run_lora_tuning


# Define some flags to specify parameters of training in shell
parser = argparse.ArgumentParser(description="Specify parameters of ViT training")
parser.add_argument("-l", "--lora", action="store_true", help="run LoRA based training")
parser.add_argument("-s", "--save", action="store_true", help="save model in output directory")
parser.add_argument("-e", "--epochs", default=5, type=int, help="specify number of epochs")
parser.add_argument("-r", "--rank", default=8, type=int, help="specify LoRA rank")
args = parser.parse_args()


def main():
    # Specify data and training parameters
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 64

    if os.path.isdir("/projects/ec232/data/"):
        # Load data from educloud server if data path exists
        print(datetime.now(), "Loading data from educloud server")
        train_loader, test_loader = get_dataloaders_educloud(BATCH_SIZE, INPUT_SHAPE)
    else:
        # Otherwise load local data from "imagewoof" directory
        print(datetime.now(), "Loading data from local path")
        train_loader, test_loader = get_dataloaders_local(BATCH_SIZE, INPUT_SHAPE)

    if not args.lora:
        # Exercise 1 - full fine-tuning
        print(datetime.now(), f"Starting full fine-tuning with {args.epochs} epochs")
        run_full_tuning(train_loader, test_loader, args.epochs, args.save)
    else:
        # Exercise 2 - LoRA approach
        print(datetime.now(), f"Starting LoRA training with {args.epochs} epochs and r={args.rank}")
        run_lora_tuning(train_loader, test_loader, args.epochs, args.rank, args.save)


if __name__ == '__main__':
    main()
