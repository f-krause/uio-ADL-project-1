import torch
import time
import timm
from timm import optim
import numpy as np
from tqdm import tqdm
from datetime import datetime


from lora import LoRATransformer

# Automatically define device
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')


def measure(fnc):
    start = time.time()
    fnc()
    end = time.time()
    print(f'Time taken: {end - start}')


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


def run_full_tuning(train_loader, test_loader, epochs: int = 5, save_model: bool = False):
    # Define model for fine-tuning
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)

    for param in model.parameters():
        param.requires_grad = False
    model.head = torch.nn.Linear(model.head.in_features, 10)
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    # measure(lambda: eval(model, test_loader, loss_fnc)) # TODO is this necessary?
    measure(lambda: train(model, train_loader, epochs, optimizer, loss_fnc))
    measure(lambda: eval(model, test_loader, loss_fnc))

    if save_model:
        save_path = "../output/full_model.pt"
        print(datetime.now(), f"Saving model to {save_path}")
        with open(save_path, "wb") as f:
            # TODO check if file already exists
            torch.save(model, f)


def run_lora_tuning(train_loader, test_loader, epochs: int = 10, save_model: bool = False):
    r = 10

    model = LoRATransformer(timm.create_model('vit_tiny_patch16_224', pretrained=True), r)
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    measure(lambda: train(model, train_loader, epochs, optimizer, loss_fnc))
    measure(lambda: eval(model, test_loader, loss_fnc))

    if save_model:
        save_path = "../output/lora_model.pt"
        print(datetime.now(), f"Saving model to {save_path}")
        with open(save_path, "wb") as f:
            # TODO check if file already exists
            torch.save(model, f)
