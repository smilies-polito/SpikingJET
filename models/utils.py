import os

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet

import tonic
import tonic.transforms as transforms



def load_ImageNet_validation_set(batch_size,
                                 image_per_class=None,
                                 imagenet_folder='~/Datasets/ImageNet'):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_validation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    validation_dataset_folder = 'tmp'
    validation_dataset_path = f'{validation_dataset_folder}/imagenet_{image_per_class}.pt'

    try:
        if image_per_class is None:
            raise FileNotFoundError

        validation_dataset = torch.load(validation_dataset_path)
        print('Resized Imagenet loaded from disk')

    except FileNotFoundError:
        validation_dataset = ImageNet(root=imagenet_folder,
                                      split='val',
                                      transform=transform_validation)

        if image_per_class is not None:
            selected_validation_list = []
            image_class_counter = [0] * 1000
            for validation_image in tqdm(validation_dataset, desc='Resizing Imagenet Dataset', colour='Yellow'):
                if image_class_counter[validation_image[1]] < image_per_class:
                    selected_validation_list.append(validation_image)
                    image_class_counter[validation_image[1]] += 1
            validation_dataset = selected_validation_list

        os.makedirs(validation_dataset_folder, exist_ok=True)
        torch.save(validation_dataset, validation_dataset_path)

    # DataLoader is used to load the dataset
    # for training
    val_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    print('Dataset loaded')

    return val_loader


def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    train_dataset = CIFAR10('weights/files/',
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10('weights/files/',
                           train=False,
                           transform=transform_test,
                           download=True)

    # If only a number of images is required per class, modify the test set
    if test_image_per_class is not None:
        image_tensors = list()
        label_tensors = list()
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                image_tensors.append(test_image[0])
                label_tensors.append(test_image[1])
                image_class_counter[test_image[1]] += 1
        test_dataset = TensorDataset(torch.stack(image_tensors),
                                     torch.tensor(label_tensors))

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                             lengths=[train_split_length, val_split_length],
                                                             generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                             batch_size=train_batch_size,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader

def load_DVSGesture_test_dataset(batch_size): 
    ### Transforms
    size = tonic.datasets.DVSGesture.sensor_size

    # Denoise transform removes outlier events with inactive surrounding pixels for 10ms
    denoise_transform = transforms.Denoise(filter_time=10000)

    # ToFrame transform bins events into 25 clusters of frames
    frame_transform = transforms.ToFrame(sensor_size=size, n_time_bins=25)

    # Chain the transforms
    all_transform = transforms.Compose([denoise_transform, frame_transform])
    test_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=False)
    cached_testset = tonic.DiskCachedDataset(test_set, cache_path='./cache/dvsgesture/test')
    test_loader = torch.utils.data.DataLoader(cached_testset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))

    return test_loader

def load_NMNIST_test_dataset(batch_size):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                        transforms.ToFrame(sensor_size=sensor_size,
                                                            n_time_bins=100)
                                        ])

    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    # no augmentations for the testset
    cached_testset = tonic.DiskCachedDataset(testset, cache_path='./data/cache/nmnist/test')
    testloader = torch.utils.data.DataLoader(cached_testset, batch_size=batch_size, shuffle=False, drop_last=True,collate_fn=tonic.collation.PadTensors(batch_first=False))

    return testloader

def load_SHD_test_dataset(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToFrame(
                sensor_size=tonic.datasets.hsd.SHD.sensor_size,
                n_time_bins=100,
            )
        ]
    )

    test_set 	= tonic.datasets.hsd.SHD(save_to='./data', train=False,transform = transform)

    test_loader 	= torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    return test_loader


def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        # state_dict = torch.load(path, map_location=device)['state_dict']
        state_dict = torch.load(path, map_location=device)
    else:
        state_dict = torch.load(path, map_location=device)

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=False)
