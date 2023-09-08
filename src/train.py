import os.path

import torch
import time
import timm
from timm import optim
import numpy as np
from tqdm import tqdm
from datetime import datetime

from lora import LoRATransformer


# Automatically define torch device
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')


def measure(fnc):
    """Measure helper function to measure training time"""
    start = time.time()
    losses = fnc()
    end = time.time()
    print(f'Time taken: {end - start}')
    return losses


def eval(model, dataloader, loss_fnc):
    """Evaluate model on test data with given loss function"""
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

    print(datetime.now(), f'Loss: {total_loss}')
    print(datetime.now(), f'Accuracy: {np.mean((pred_targets == pred_outputs))}')


def train(model, dataloader, epochs, optimizer, loss_fnc):
    """Train model on train data for given optimizer and loss function"""
    model.train()
    losses = []
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
        losses.append(total_loss)
        print(datetime.now(), f'Epoch: {epoch + 1}, loss: {total_loss}')

    return losses


def save_data(model_name, model, losses):
    """Helper function to save models and losses as well as prevent overwriting"""
    save_path_model = f"/output/{model_name}_model.pt"
    save_path_csv = f"/output/losses_{model_name}_model.csv"
    if os.path.isfile(save_path_model):
        # If model is already saved, rename files to not overwrite existing files
        save_path_model = save_path_model[:-3] + "_" + str(datetime.now())[-5:] + ".pt"
        save_path_csv = save_path_csv[:-4] + "_" + str(datetime.now())[-5:] + ".csv"
    print(datetime.now(), f"Saving model to {save_path_model}")
    torch.save(model.state_dict(), save_path_model)  # save state dict of model
    np.savetxt(save_path_csv, np.array(losses), delimiter=",")  # save losses as csv


def run_full_tuning(train_loader, test_loader, epochs, save_model: bool = False):
    """Helper function to run full fine-tuning"""
    # Define model for fine-tuning
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)

    for param in model.parameters():
        param.requires_grad = False
    model.head = torch.nn.Linear(model.head.in_features, 10)  # replace old model head # FIXME test if necessary
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    losses = measure(lambda: train(model, train_loader, epochs, optimizer, loss_fnc))  # run model training
    eval(model, test_loader, loss_fnc)  # run evaluation of trained model

    if save_model:
        save_data("full", model, losses)


def run_lora_tuning(train_loader, test_loader, epochs, r, save_model: bool = False):
    """Helper function to run LoRA based training"""
    model = LoRATransformer(timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10), r)  # FIXME test if num_classes argument should be removed
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    losses = measure(lambda: train(model, train_loader, epochs, optimizer, loss_fnc))  # run model training
    eval(model, test_loader, loss_fnc)  # run evaluation of trained model

    if save_model:
        save_data(f"lora_r{r}", model, losses)
