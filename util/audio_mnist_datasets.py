import os
import random
import urllib
import pickle

from torch.utils.data import Dataset
from scipy.io.wavfile import read

"""
Class for an Audio-MNIST dataset. More or less the same as other datasets used
throughout this project.
"""
class AudioMNISTDataset(Dataset):

    """
    Constructor for an Audio-MNIST dataset. Uses pickling to load data.

    @param root      the root directory of the pickled data
    @param name      the name of the model (optional)
    @param train     whether this is a training or validation dataset
    @param transform any transforms applied to this data [TODO: research audio transforms]
    @param encoded   whether the data is encoded or not [TODO: find out if this is actually necessary]
    """
    def __init__(self, root, name='', train=True, transform=None, encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            self.data_path = os.path.join(root, 'train.pickle') # Get the training data
        else:
            self.data_path = os.path.join(root, 'val.pickle')   # Get the validation data
        self.data = pickle.load(self.data_path)
        self.encoded = encoded # still not sure if I need this, just copying it over from LMDB dataset for now

    """
    @param index  the index of the dataset from which to retrieve an item
    @return       the data and label at the given index
    """
    def __getitem__(self, index):
        return self.data[index], 0 # might be able to add labels but don't see the need if we're making a generative model
        
    """
    @return  the length of the data
    """
    def __len__(self):
        return self.data.shape[0]

"""
Download the Audio-MNIST dataset from GitHub, pickle it and save it to
the given location. Thanks to soerenab and their repository:
https://github.com/soerenab/AudioMNIST/

@param data_dir  the location in which to save the data
"""
def load_audio_mnist_dataset(data_dir):
    # Make the directory in which to store the data if it doesn't exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print('Created directory', data_dir)
    else:
        print(data_dir, 'already exists.')
    train, val = split_data()
    download_files(train, os.path.join(data_dir, 'train.pickle'))
    download_files(val, os.path.join(data_dir, 'val.pickle'))
    return

"""
Download the files corresponding to the relevant batch (either training or validation).

@param batches   the batches of data to download (this will vary depending on whether
                 it is training or validation data)
@param save_dir  the location of the pickle file that will be written in this method
"""
def download_files(batches, save_dir):
    # This is the base of the Git repo containing all the audio-MNIST data
    url_base = 'https://github.com/soerenab/AudioMNIST/blob/master/data/'
    data = []
    starting_batch = 0
    if os.path.exists(save_dir):
        # We already have some (or all) of the data pickled, so start where
        # we left off
        with open(save_dir, 'r') as pickle_file:
            data = pickle.load(pickle_file)
        # Each batch is of length 50 so we can just start from whichever batch
        # we last saved
        starting_batch = len(data) // 50
    for b in batches[starting_batch:]:
        print(f'Preparing to download batch {b} to file {save_dir}.')
        # Filesystem stores batches less than 10 as eg. '09'
        batch = '0' + str(b) if b < 10 else str(b)
        batch_dir = os.path.join(url_base, batch)
        for i in range(10):
            for j in range(50):
                filename = f'{i}_{batch}_{j}'
                # Need to get the raw data from GitHub
                filepath = os.path.join(batch_dir, f'{filename}.wav?raw=true')
                read_data = grab_data(filepath)
                data.append(read_data)

            print(f'Downloaded all utterances of {i} in batch {b}...')
        print(f'Downloaded batch {b}.')
        # Writes the total pickle file batch-by-batch so you can start from
        # where you left off if you need to stop downloading for any reason
        with open(save_dir, 'wb') as savefile:
            pickle.dump(data, savefile)
    
"""
Grab an individual WAV file from the Audio-MNIST dataset.

@param filepath  the URL containing the raw data on GitHub
@return          the WAV file as a numpy array, or another call to this function
                 if it encountered a HTTPError (normally a 502 error)
"""
def grab_data(filepath):
    # Sometimes you can get a 502 error which can stop the whole downloading
    # process. This recursive try-catch function is designed to continually
    # ping the server until it finally gets a packet.
    try:
        wav_data, _ = urllib.request.urlretrieve(filepath)
        # scipy.io.wavfile will read the raw data and convert it to an array
        rate, read_data = read(wav_data)
        return read_data
    except urllib.error.HTTPError:
        return grab_data(filepath)

"""
Split the Audio-MNIST data on GitHub according to a random seed and a train-validation
split.

@param seed              the random seed with which to split the data, default is 255
@param train_percentage  the percentage of the data that should go towards training
@return                  a tuple of lists containing the indices of the batches
                         corresponding to training and validation sets, respectively
"""
def split_data(seed=255, train_percentage=0.9): # Default value is a random integer chosen from RANDOM.org (coincidentally 255!)
    random.seed(seed) 
    train_val_split = ([], []) # first element is training, second element is validation
    num_batches = 60 # this is the number of batches in the data
    distribution = random.choices([0, 1], weights=[train_percentage, 1 - train_percentage], k=num_batches)
    for i in range(1, num_batches + 1):
        train_val_split[distribution[i - 1]].append(i)
    return train_val_split