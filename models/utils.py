import os

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

import tonic
import tonic.transforms as tonic_transforms



def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),   # Normalize the data (stability for training)
    ])

    train_dataset = CIFAR10('./data',
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10('./data',
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
    denoise_transform = tonic_transforms.Denoise(filter_time=10000)

    # ToFrame transform bins events into 25 clusters of frames
    frame_transform = tonic_transforms.ToFrame(sensor_size=size, n_time_bins=25)

    # Chain the transforms
    all_transform = transforms.Compose([denoise_transform, frame_transform])
    # tonic's figshare link is refused to scripts: fetch through the figshare API instead
    from download_datasets import ensure_dvsgesture
    ensure_dvsgesture('./data', train=False)
    test_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=False)
    cached_testset = tonic.DiskCachedDataset(test_set, cache_path='./data/cache/dvsgesture/test')
    test_loader = torch.utils.data.DataLoader(cached_testset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))

    return test_loader

def load_NMNIST_test_dataset(batch_size):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = tonic_transforms.Compose([tonic_transforms.Denoise(filter_time=10000),
                                        tonic_transforms.ToFrame(sensor_size=sensor_size,
                                                            n_time_bins=100)
                                        ])

    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    # no augmentations for the testset
    cached_testset = tonic.DiskCachedDataset(testset, cache_path='./data/cache/nmnist/test')
    testloader = torch.utils.data.DataLoader(cached_testset, batch_size=batch_size, shuffle=False, drop_last=True,collate_fn=tonic.collation.PadTensors(batch_first=False))

    return testloader

def load_from_dict(network, device, path, function=None):
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:     # {'state_dict': ...} or a raw state dict
        state_dict = state_dict['state_dict']

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=True)          # a checkpoint that does not fit fails loudly
