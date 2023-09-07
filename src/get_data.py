import torchvision
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import litdata


def get_dataloader(path: str, shuffle: bool, batch_size: int, input_shape: list):
    """Build dataloader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_shape[:2]),
    ])
    dataset_folder = torchvision.datasets.DatasetFolder(
        root=path,
        loader=lambda p: Image.open(p).convert('RGB'),
        transform=transform,
        is_valid_file=lambda x: True
    )
    # Shuffle only once testing data.
    if shuffle:
        np.random.shuffle(dataset_folder.samples)
    data_loader = torch.utils.data.DataLoader(
        dataset_folder, batch_size=batch_size, shuffle=shuffle
    )
    return data_loader


def get_dataloaders_local(batch_size: int, input_shape: list):
    """Get train and test data loader based on local data"""
    train_loader = get_dataloader(
        path="../imagewoof/train",
        shuffle=True,
        batch_size=batch_size,
        input_shape=input_shape
    )
    test_loader = get_dataloader(
        path="../imagewoof/test",
        shuffle=False,
        batch_size=batch_size,
        input_shape=input_shape
    )

    return train_loader, test_loader


class ToRGBTensor:
    """Code from Mariuaas copied from Discourse"""

    def __call__(self, img):
        return transforms.functional.to_tensor(img).expand(3, -1, -1)  # Expand to 3 channels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_dataloaders_educloud(batch_size: int, input_shape: list):
    """
    Get train and test data loader based on educload .tar data
    Code adapted from Mariuaas from Discourse
    """

    # Define data path
    DATA_PATH = '/projects/ec232/data/'

    # Define mean and std from ImageNet data
    IN_MEAN = [0.485, 0.456, 0.406]
    IN_STD = [0.229, 0.224, 0.225]

    # Define postprocessing / transform of data modalities
    postprocess = (  # Create tuple for image and class...
        transforms.Compose([  # Handles processing of the .jpg image
            ToRGBTensor(),  # Convert from PIL image to RGB torch.Tensor.
            transforms.Resize(input_shape[:2]),  # Resize images
            transforms.Normalize(IN_MEAN, IN_STD),  # Normalize image to correct mean/std.
        ]),
        torch.nn.Identity(),  # Handles proc. of .cls file (just an int).
    )

    # Load training and validation data
    train_data = litdata.LITDataset('ImageWoof', DATA_PATH).map_tuple(*postprocess)
    test_data = litdata.LITDataset('ImageWoof', DATA_PATH, train=False).map_tuple(*postprocess)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
