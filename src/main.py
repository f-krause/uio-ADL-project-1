import torch
import torchvision
import numpy as np
import cv2
import timm
import time

from tqdm import tqdm
from torchvision import transforms
from timm import optim


def get_data_loader(path: str, transforms, shuffle: bool, batch_size: int) -> torch.utils.data.DataLoader:
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
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fnc(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, loss: {total_loss}')


def eval(model, dataloader):
    model.eval()
    total_loss = 0
    pred_targets = np.array([])
    pred_outputs = np.array([])
    for batch in tqdm(dataloader):
        inputs, targets = batch
        outputs = model(inputs)
        pred_targets = np.hstack([pred_targets, targets.numpy()])
        pred_outputs = np.hstack([pred_outputs, np.argmax(outputs.detach().numpy(), axis=1)])
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
    INPUT_SHAPE = (224, 224, 3)
    train_loader = get_data_loader("../imagewoof/train",
                                   transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Resize(INPUT_SHAPE[:2]),
                                       transforms.Normalize([0, 0, 0], [255, 255, 255])
                                   ]),
                                   False,
                                   4)
    test_loader = get_data_loader("../imagewoof/train",
                                  transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize(INPUT_SHAPE[:2]),
                                      transforms.Normalize([0, 0, 0], [255, 255, 255])
                                  ]),
                                  True,
                                  4)

    model = timm.create_model('vit_tiny_patch16_224', num_classes=10, pretrained=True)
    optimizer = timm.optim.AdamP(model.parameters(), lr=0.01)
    loss_fnc = torch.nn.CrossEntropyLoss()

    # train(
    #     model,
    #     train_loader,
    #     5,
    #     optimizer,
    #     loss_fnc
    # )

    measure(lambda: eval(
        model,
        test_loader
    ))
