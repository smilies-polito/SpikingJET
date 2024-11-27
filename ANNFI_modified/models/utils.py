import os

from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import TensorDataset, Dataset, random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet

import tonic
import tonic.transforms as transforms
from types import SimpleNamespace
from models.dataset import DVS_Lip
import matplotlib.pyplot as plt
import time

import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn


from torch.utils.data import DataLoader


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

def load_audiomnist_dataset(batch_size):
    root_dir = "./data/audiomnist"
    batch_size = batch_size
    sample_rate = 48e3
    fft_window = 25e-3
    hop_length_s = 10e-3
    n_mels = 40
    spiking_thresh = 0.9

    transform = MelFilterbank(
        sample_rate=sample_rate,
        fft_window=fft_window,
        hop_length_s=hop_length_s,
        n_mels=n_mels,
        db=True,
        normalize=True,
        spikify=True,
        spiking_thresh=spiking_thresh
    )

    dataset = CustomDataset(root_dir=root_dir, transform=transform)
    # train_size = int(0.7 * len(dataset))
    # validation_size = int(0.2 * len(dataset))
    # test_size = len(dataset) - train_size - validation_size

    # train_set, validation_set, test_set = random_split(dataset, [train_size, validation_size, test_size])

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    return test_loader



def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    print("bs " + str(batch_size))
    print("nw " + str(num_workers))
    loader =  DataLoader(dataset,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         shuffle = shuffle,
                         drop_last = False,
                         pin_memory=True)
    return loader

def load_DVS_Lip_test_dataset(batch_size):
    args = SimpleNamespace(**{
        'gpus': '0',
        'lr': 0.0003,
        'batch_size': 32,
        'n_class': 100,
        'seq_len': 30,
        'num_workers': 12,
        'max_epoch': 80,
        'num_bins': [1, 7],
        'test': True,
        'log_dir': None,
        'weights': 'mstp',
        'event_root': './data/DVS-Lip',
        'se': False,
        'base_channel': 64,
        'alpha': 4,
        'beta': 7,
        't2s_mul': 2
    })  
    dataset = DVS_Lip('test', args)
    print(f'Start Testing, Data Length: {len(dataset)}')
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)
    return loader

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







# AUDIOMNIST UTILITIES

class CustomDataset(Dataset):

	def __init__(self, root_dir, transform=None, max_length = 35000):

		"""
		Args:
			root_dir	: str. Directory containing
					subdirectories, one for each user

			transform	: callable, optional. Transform to be
					applied on a sample.
		"""
		self.root_dir 	= root_dir
		self.transform	= transform
		self.max_length	= max_length

		self.data = []

		# Loop over all the users' directories
		for user_folder in os.listdir(root_dir):
			#print("user_folder",user_folder)
			user_path = os.path.join(root_dir, user_folder)
			#print("user_path",user_path)
			if os.path.isdir(user_path):
				#print("a")
				# Loop over all the WAV recordings
				for file_name in os.listdir(user_path):
					#print("file_name",file_name)
					if file_name.endswith(".wav"):

						file_path = os.path.join(
							user_path,
							file_name
						)
						#print("file_path",file_path)
						# Extract label from filename
						label = int(
							file_name.split("_")[0]
						)
						#print("label",label)
						self.data.append(
							(file_path, label)
						)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		file_path, label = self.data[idx]
		#print(file_path)
		waveform, sample_rate = torchaudio.load(file_path)
		#print("waveform.shape", waveform.shape)
		#print("waveform[0].shape", waveform[0].shape)
		#print("waveform[0]", waveform[0])
		#print(sample_rate)
		# Pad or truncate the waveform to match max_length
		if waveform.size(1) > self.max_length:
			waveform = waveform[:, :self.max_length]

		elif waveform.size(1) < self.max_length:
			pad_size = self.max_length - waveform.size(1)
			waveform = fn.pad(waveform, (0, pad_size))

		if self.transform:
			waveform = self.transform(waveform)

		# --- If converting to snnTorch the part under this can be
		# modified ---

		# Reshape and return lens to make it compatible with sparch
		waveform = waveform.squeeze(dim=0).permute(1, 0)

		lens = torch.tensor([x.shape[0] for x in waveform])

		return waveform, label 
        #return waveform, lens, label

class MelFilterbank:

	def __init__(self, sample_rate = 48e3, fft_window = 25e-3,
		hop_length_s = 10e-3, n_mels = 40, db = False, normalize = False,
		spikify = False, spiking_thresh = 0.9):

		self.sample_rate	= sample_rate
		self.n_fft		= int(fft_window * sample_rate)
		self.hop_length		= int(hop_length_s * sample_rate)
		self.n_mels		= n_mels

		self.db			= db

		if self.db:
			# Convert the Mel Spectrogram to dB scale
			self.db_transform = torchaudio.transforms.\
						AmplitudeToDB()

		self.normalize		= normalize
		self.spikify		= spikify
		self.spiking_thresh	= spiking_thresh

		# Define the MelSpectrogram transform
		self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
			sample_rate	= self.sample_rate,
			n_fft		= self.n_fft,
			hop_length	= self.hop_length,
			n_mels		= self.n_mels
		)


	def __call__(self, waveform):

		# Apply the Mel Spectrogram transform
		mel_spec = self.mel_spectrogram(waveform)

		if self.db:
			# Convert the Mel Spectrogram to dB scale
			mel_spec = self.db_transform(mel_spec)

		if self.normalize:
			# Normalize mel spectrogram
			mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

		if self.spikify:
			# Convert spectrogram into spike trains
			mel_spec = (mel_spec > self.spiking_thresh).float()

		return mel_spec
