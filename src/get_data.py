import torchvision
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import litdata


def get_dataloader_local(path: str, transforms, shuffle: bool, batch_size: int):
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


class ToRGBTensor:
    """Code from Mariuaas copied from Discourse"""

    def __call__(self, img):
        return F.to_tensor(img).expand(3, -1, -1)  # Expand to 3 channels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_dataloader_educloud(batch_size: int):
    """Code adapted from Mariuaas from Discourse"""

    datapath = '/projects/ec232/data/'

    # Define mean and std from ImageNet data
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    # Define postprocessing / transform of data modalities
    postprocess = (  # Create tuple for image and class...
        transforms.Compose([  # Handles processing of the .jpg image
            ToRGBTensor(),  # Convert from PIL image to RGB torch.Tensor.
            transforms.Resize((224, 224)),  # Resize images
            transforms.Normalize(in_mean, in_std),  # Normalize image to correct mean/std.
        ]),
        torch.nn.Identity(),  # Handles proc. of .cls file (just an int).
    )

    # Load training and validation data
    traindata = litdata.LITDataset('ImageWoof', datapath).map_tuple(*postprocess)
    testdata = litdata.LITDataset('ImageWoof', datapath, train=False).map_tuple(*postprocess)

    data_loader_train = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=False
    )

    return data_loader_train, data_loader_test
