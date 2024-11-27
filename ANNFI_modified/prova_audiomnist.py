import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fn
import snntorch as snn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

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

		return waveform, lens, label

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


class Readout(snn.Leaky):

	def __init__(self, beta):

		super().__init__(beta, reset_mechanism = "none")

	def forward(self, cur, mem):

		_, mem = super().forward(cur, mem)

		return mem, fn.softmax(mem, dim = 1)


class Net(nn.Module):
	def __init__(self, num_inputs = 40, num_hidden = 128,num_outputs = 10, model = "lif", bias = False,beta = 0.9, threshold = 1.):
		super().__init__()
		self.beta1=torch.ones(num_hidden)*beta
		self.beta2=torch.ones(num_hidden)*beta
		self.beta3=torch.ones(num_outputs)*beta

		self.th1=torch.ones(num_hidden)
		self.th2=torch.ones(num_hidden)
		self.th3=torch.ones(num_outputs)

		self.fc1 = nn.Linear(in_features	= num_inputs,out_features = num_hidden,bias = bias,)
		self.lif1 = snn.Leaky(beta = self.beta1,threshold = self.th1,reset_mechanism="subtract",learn_beta=True,learn_threshold=True)

		self.fc2 = nn.Linear(in_features	= num_hidden,out_features	= num_hidden, bias = bias,)
		self.lif2 = snn.Leaky(beta = self.beta2,threshold = self.th2,reset_mechanism="subtract",learn_beta=True,learn_threshold=True)

		self.fc3 = nn.Linear(in_features	= num_hidden,out_features	= num_outputs, bias = bias,)
		self.lif3 = snn.Leaky(beta=self.beta3, threshold=self.th3, reset_mechanism="none", 
							learn_beta=True, learn_threshold=True)
		#self.readout = Readout(beta	= beta)
	
	def forward(self, input_spikes):
		mem1 = self.lif1.reset_mem()
		mem2 = self.lif2.reset_mem()
		mem3 = self.lif3.reset_mem()
		mem3_rec = []
		spk3_rec = []
		#print(input_spikes.shape[0])
		for step in range(input_spikes.shape[0]):
			#print(input_spikes[step].shape)
			cur1 = self.fc1(input_spikes[step])
			spk1, mem1 = self.lif1(cur1, mem1)

			cur2 = self.fc2(spk1)
			spk2, mem2 = self.lif2(cur2, mem2)

			cur3 = self.fc3(spk2)
			spk3,mem3=self.lif3(cur3,mem3)

			mem3_rec.append(mem3)
			spk3_rec.append(spk3)
		return torch.stack(mem3_rec, dim=0), \
            torch.stack(spk3_rec, dim=0)

def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        # state_dict = torch.load(path, map_location=device)['state_dict']
        state_dict = torch.load(path, map_location=device)
    else:
        state_dict = torch.load(path, map_location=device)

    if function is None:
        clean_state_dict = {
            key.replace('module.', ''): value 
            for key, value in state_dict.items()
        }
    else:
        clean_state_dict = {
            key.replace('module.', ''): function(value) 
            if not (('bn' in key) and ('weight' in key)) else value 
            for key, value in state_dict.items()
        }

    network.load_state_dict(clean_state_dict, strict=False)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    start_time = time.time()

    root_dir = "./data/audiomnist"
    batch_size = 32
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
   
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    num_inputs = n_mels
    num_hidden = 150
    num_outputs = 10
    beta = 0.9375
    trained_param_dir = "./TrainedParameters"
    trained_param_dict = trained_param_dir + "/state_dict_audiomnist_snn_memavg_1.pt"

    print("Creating the network\n")
    net = Net(
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        beta=beta
    )

    loss_fn = nn.CrossEntropyLoss()
    load_from_dict(network=net, device=device, path=trained_param_dict)

    loss_hist = []
    validation_loss_hist = []
    test_loss_hist = []
    net.to(device)

    with torch.no_grad():
        net.eval()
        for _, (test_data, _, test_labels) in enumerate(test_loader):
            test_data = test_data.permute(1, 0, 2).to(device)
            test_labels = test_labels.to(device)
            mem_rec, spk_rec = net(test_data)
            avg_mem = mem_rec.mean(dim=0)
            pro = torch.softmax(avg_mem, dim=-1)
            test_loss_val = loss_fn(pro, test_labels)

        _, idx = avg_mem.max(1)
        test_acc = np.mean((test_labels == idx).detach().cpu().numpy())

    print(f"Test Set Loss: {test_loss_val:.2f}")
    print(f"Test set accuracy: {test_acc * 100:.2f}%")

    torch.save(net.state_dict(), trained_param_dict)

    end_time = time.time()
    print("Total required time: ", end_time - start_time)
