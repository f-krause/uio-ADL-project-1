import torch
import torchvision
import numpy as np
from PIL import Image
import timm
import time
import os

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
    dataset_folder = torchvision.datasets.DatasetFolder(
        root=path,
        loader=lambda p: Image.open(p).convert('RGB'),
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


def get_dataloader_educloud(batch_size: int):
    # Import necessary packages
    import litdata
    from torch import nn
    import torchvision.transforms as T

    # Specify data folder
    datapath = '/projects/ec232/data/'

    # Define mean and std from ImageNet data
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    # Define postprocessing / transform of data modalities
    postprocess = (  # Create tuple for image and class...
        T.Compose([  # Handles processing of the .jpg image
            T.ToTensor(),  # Convert from PIL image to torch.Tensor
            T.Normalize(in_mean, in_std),  # Normalize image to correct mean/std.
        ]),
        nn.Identity(),  # Handles proc. of .cls file (just an int).
    )

    # Load training and validation data
    traindata = litdata.LITDataset('ImageWoof', datapath).map_tuple(*postprocess)
    testdata = litdata.LITDataset('ImageWoof', datapath, train=False).map_tuple(*postprocess)

    print("###############")
    print(traindata)
    print("type", type(traindata))
    print("###############")

    data_loader_train = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=False
    )

    return data_loader_train, data_loader_test


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
    BATCH_SIZE = 64

    if os.path.isdir("/projects/ec232/data/"):
        # TODO read from tar files
        train_loader, test_loader = get_dataloader_educloud(BATCH_SIZE)

    else:
        train_loader = get_dataloader(
            "../imagewoof/train",
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(INPUT_SHAPE[:2]),
            ]),
            False,
            BATCH_SIZE
        )
        test_loader = get_dataloader(
            "../imagewoof/train",
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(INPUT_SHAPE[:2]),
            ]),
            True,
            BATCH_SIZE
        )

    # Exercise 1:
    # Define model for fine-tuning
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
    for param in model.parameters():
        param.requires_grad = False
    model.head = torch.nn.Linear(model.head.in_features, 10)
    model = model.to(device)
    optimizer = timm.optim.AdamW(model.parameters())
    loss_fnc = torch.nn.CrossEntropyLoss()

    measure(lambda: eval(model, test_loader, loss_fnc))
    measure(lambda: train(model, train_loader, 5, optimizer, loss_fnc))
    measure(lambda: eval(model, test_loader, loss_fnc))

    # Exercise 2:
    # r = 10
    # model = LoRATransformer(timm.create_model('vit_tiny_patch16_224', pretrained=True), r)  # TODO also add ", num_classes=10" ?
    # model = model.to(device)
    # optimizer = timm.optim.AdamW(model.parameters())
    # loss_fnc = torch.nn.CrossEntropyLoss()
    #
    # measure(lambda: train(model, train_loader, 10, optimizer, loss_fnc))
    # measure(lambda: eval(model, test_loader, loss_fnc))

