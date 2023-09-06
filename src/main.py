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

# Automatically define device
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')


def train(model, dataloader, epochs, optimizer, loss_fnc):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()

            # Load data onto device
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Predictions
            outputs = model(inputs)
            loss = loss_fnc(outputs, targets)

            # Optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch: {epoch + 1}, loss: {total_loss}')


def eval(model, dataloader, loss_fnc):
    model.eval()
    total_loss = 0
    pred_targets = np.array([])
    pred_outputs = np.array([])

    for batch in tqdm(dataloader):
        # Load data onto device
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Predictions
        outputs = model(inputs)
        pred_targets = np.hstack([pred_targets, targets.cpu().numpy()])
        pred_outputs = np.hstack([pred_outputs, np.argmax(outputs.cpu().detach().numpy(), axis=1)])
        loss = loss_fnc(outputs, targets)
        total_loss += loss.item()

    print(f'Loss: {total_loss}')
    print(f'Accuracy: {np.mean((pred_targets == pred_outputs))}')


def measure(fnc):
    start = time.time()
    fnc()
    end = time.time()
    print(f'Time taken: {end - start}')


def main():
    # Define dataloaders
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 64

    if os.path.isdir("/projects/ec232/data/"):
        # Load data from educloud server if path exists
        train_loader, test_loader = get_dataloaders_educloud(BATCH_SIZE, INPUT_SHAPE)

    else:
        # Otherwise load local data
        train_loader, test_loader = get_dataloaders_local(BATCH_SIZE, INPUT_SHAPE)


    # Exercise 1 - full fine-tuning
    # Define model for fine-tuning
    """
    NR_EPOCHS = 5
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
    for param in model.parameters():
        param.requires_grad = False
    model.head = torch.nn.Linear(model.head.in_features, 10)
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    # measure(lambda: eval(model, test_loader, loss_fnc)) # TODO is this necessary?
    measure(lambda: train(model, train_loader, NR_EPOCHS, optimizer, loss_fnc))
    measure(lambda: eval(model, test_loader, loss_fnc))
    """

    # Exercise 2 - LoRA approach
    r = 10
    NR_EPOCHS = 10
    model = LoRATransformer(timm.create_model('vit_tiny_patch16_224', pretrained=True), r)
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    measure(lambda: train(model, train_loader, NR_EPOCHS, optimizer, loss_fnc))
    measure(lambda: eval(model, test_loader, loss_fnc))


if __name__ == '__main__':
    main()
