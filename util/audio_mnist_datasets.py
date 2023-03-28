import os
import random
import urllib
import pickle

from torch.utils.data import Dataset
from scipy.io.wavfile import read

class AudioMNISTDataset(Dataset):

    def __init__(self, root, name='', train=True, transform=None, encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            self.data_path = os.path.join(root, 'train.pickle')
        else:
            self.data_path = os.path.join(root, 'val.pickle')
        self.data = load(self.data_path)
        self.encoded = encoded # still not sure if I need this, just copying it over from LMDB dataset for now

    def __getitem__(self, index):
        return self.data[index], 0 # might be able to add labels but don't see the need if we're making a generative model
        
    def __len__(self):
        return self.data.shape[0]


def load_audio_mnist_dataset(data_dir):
    # Make all the directories necessary for the dataset
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print('Created directory', data_dir)
    else:
        print(data_dir, 'already exists.')
    train, val = split_data()
    download_files(train, os.path.join(data_dir, 'train.pickle'))
    download_files(val, os.path.join(data_dir, 'val.pickle'))
    return

def download_files(batches, save_dir):
    # This is the base of the Git repo containing all the audio-MNIST data
    url_base = 'https://github.com/soerenab/AudioMNIST/blob/master/data/'
    data = []
    starting_batch = 0
    if os.path.exists(save_dir):
        with open(save_dir, 'r') as pickle_file:
            data = pickle.load(pickle_file)
        starting_batch = len(data) // 50
    for b in batches[starting_batch:]:
        print(f'Preparing to download batch {b} to file {save_dir}.')
        batch = '0' + str(b) if b < 10 else str(b)
        batch_dir = os.path.join(url_base, batch)
        for i in range(10):
            for j in range(50):
                filename = f'{i}_{batch}_{j}'
                filepath = os.path.join(batch_dir, f'{filename}.wav?raw=true')
                read_data = grab_data(filepath)
                data.append(read_data)

            print(f'Downloaded all utterances of {i} in batch {b}...')
        print(f'Downloaded batch {b}.')
        with open(save_dir, 'wb') as savefile:
            pickle.dump(data, savefile)
    

def grab_data(filepath):
    try:
        wav_data, _ = urllib.request.urlretrieve(filepath)
        rate, read_data = read(wav_data)
        return read_data
    except urllib.error.HTTPError:
        return grab_data(filepath)


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