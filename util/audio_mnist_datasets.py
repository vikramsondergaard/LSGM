import os
import random

from torch.utils.data import Dataset
from numpy import array
from pickle import load, dump
from urllib.request import urlretrieve
from scipy.io.wavfile import read

class AudioMNISTDataset(Dataset):

    def __init__(self, root, name='', train=True, transform=None, encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            self.data_path = os.path.join(root, 'audio-mnist/train/')
        else:
            self.data_path = os.path.join(root, 'audio-mnist/val/')
        self.data = array([load(os.path.join(self.data_path, f)) for f in os.listdir(self.data_path)])
        self.encoded = encoded # still not sure if I need this, just copying it over from LMDB dataset for now

    def __getitem__(self, index):
        return self.data[index], 0 # might be able to add labels but don't see the need if we're making a generative model
        
    def __len__(self):
        return self.data.size[0]

def download_audio_mnist_dataset(data_dir):
    # Make all the directories necessary for the dataset
    make_directories(data_dir)
    train, val = split_data()
    download_files(train, os.path.join(data_dir, 'train'))
    download_files(val, os.path.join(data_dir, 'val'))
    return

def make_directories(data_dir):
    # Make the initial directory if it doesn't already exist
    make_directory(data_dir)
    # Split the data according to the default random seed and train percentage
    train, val = split_data()
    # Make the training directory if it doesn't already exist
    train_dir = os.path.join(data_dir, 'train')
    make_directory(train_dir)
    # Make the validation directory if it doesn't already exist
    val_dir = os.path.join(data_dir, 'val')
    make_directory(val_dir)

def make_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print('Created directory', directory)
    else:
        print(directory, 'already exists.')

def download_files(batches, save_dir):
    # This is the base of the Git repo containing all the audio-MNIST data
    url_base = 'https://github.com/soerenab/AudioMNIST/tree/master/data'
    for b in batches:
        print(f'Preparing to download batch {b}')
        batch = '0' + str(b) if b < 10 else str(b)
        batch_dir = os.path.join(url_base, batch)
        for i in range(10):
            for j in range(50):
                filename = f'{i}_{batch}_{j}'
                if not os.path.exists(os.path.join(save_dir, f'{filename}.pickle')):
                    file = os.path.join(batch_dir, f'{filename}.wav?raw=true')
                    data, _ = urlretrieve(file)
                    save_path = os.path.join(save_dir, f'{filename}.pickle')
                    with open(save_path, 'wb') as savefile:
                        dump(data, savefile)
            print(f'Downloaded all utterances of {i} in batch {b}...')
        print(f'Downloaded batch {b}.')


def split_data(seed=255, train_percentage=0.9): # Default value is a random integer chosen from RANDOM.org (coincidentally 255!)
    random.seed(seed) 
    train_val_split = ([], []) # first element is training, second element is validation
    num_batches = 60
    distribution = random.choices([0, 1], weights=[train_percentage, 1 - train_percentage], k=num_batches)
    for i in range(1, num_batches + 1):
        train_val_split[distribution[i - 1]].append(i)
    return train_val_split

def pickle_files(data_dir, is_train, batches):
    if is_train:
        save_dir = os.path.join(data_dir, 'train')
    else:
        save_dir = os.path.join(data_dir, 'val')

    dataset_dir = os.path.join(data_dir, 'data')

    for b in batches:
        batch_name = '0' + str(b) if b < 10 else str(b) # batch numbers less than 10 are eg. '09'
        batch_dir = os.path.join(dataset_dir, batch_name)

    for batch in [f for f in os.listdir() if f.isnumeric()]:
    # Get every WAV file inside every batch directory
        for file in os.listdir(f'./{batch}'):
            rate, wav_data = read(f'./{batch}/{file}')
            # Remove the WAV file extension and replace with the pickle extension
            filename = file[:-3] + 'pickle'
            num = int(file[2:4])
            dest = f'./train/{filename}' if num in train else f'./val/{filename}'
            with open(dest, 'wb') as pickle_file:
                dump(wav_data, pickle_file)