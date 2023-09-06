import torch
import torchvision
import numpy as np
import cv2
import timm
import time

from tqdm import tqdm
from torchvision import transforms
from timm import optim
from lora import LoRATransformer


# Define device
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')


def get_dataloader(path: str, transforms, shuffle: bool, batch_size: int):
    root_path = path
    transforms = transforms

    dataset_folder = torchvision.datasets.DatasetFolder(
        root=root_path,
        loader=lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
        transform=transforms,
        is_valid_file=lambda x: True
    )
    # Shuffle only once testing data.
    if shuffle:
        np.random.shuffle(dataset_folder.samples)
    data_loader = torch.utils.data.DataLoader(
        dataset_folder, batch_size=batch_size, shuffle=shuffle
    )
    return data_loader


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

        print(f'Epoch: {epoch}, loss: {total_loss}')


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


if __name__ == '__main__':
    # Define dataloaders
    INPUT_SHAPE = (224, 224, 3)
    train_loader = get_dataloader(
        "../imagewoof/train",
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(INPUT_SHAPE[:2]),
        ]),
        False,
        64
    )
    test_loader = get_dataloader(
        "../imagewoof/train",
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(INPUT_SHAPE[:2]),
        ]),
        True,
        64
    )

    # Exercise 1:
    # # Define model for fine-tuning
    # model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.head = torch.nn.Linear(model.head.in_features, 10)
    # model = model.to(device)
    # optimizer = timm.optim.AdamW(model.parameters())
    # loss_fnc = torch.nn.CrossEntropyLoss()
    #
    # measure(lambda: eval(model, test_loader, loss_fnc))
    # measure(lambda: train(model, train_loader, 5, optimizer, loss_fnc))
    # measure(lambda: eval(model, test_loader, loss_fnc))

    # Exercise 2:
    r = 10
    model = LoRATransformer(timm.create_model('vit_tiny_patch16_224', pretrained=True), r)
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    measure(lambda: train(model, train_loader, 10, optimizer, loss_fnc))
    measure(lambda: eval(model, test_loader, loss_fnc))

