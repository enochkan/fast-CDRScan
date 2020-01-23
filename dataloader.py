from customloader import CustomLoader
import numpy as np
import math
# import torch
# from torch.utils.data import Dataset, DataLoader
from utils import to_binary, smiles_to_onehot
import sys
sys.dont_write_bytecode = True
np.set_printoptions(threshold=sys.maxsize)

import keras

class CDRDataGenerator(keras.utils.Sequence):
    """Cosmic Cell-line Project Dataset."""

    def __init__(self, dataset, data_dir, list_IDs, batch_size=1, dim=(32,32,32), 
    shuffle=True, transform=None, verbose=0, prows=None):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data = dataset
        self.transform = transform
        self.verbose = verbose
        self.X_gene = to_binary(self.data)
        self.X_chem = smiles_to_onehot(self.data)
        self.Y = self.data['LN_IC50'].tolist()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size), dtype=float)


        if self.verbose:
            print('Loaded 1D genomic data with shape... ' + str(self.X_gene.shape))
            print('Loaded 1D chemical data with shape... ' + str(self.X_chem.shape))
            print('Loaded IC50 data with shape... ' + str(len(self.Y)))

        # Generate data 
        # mu, sigma = 0, 1
        # noise = np.random.normal(mu, sigma, size=(self.dim, 1))
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # zeros_gene = np.ones(shape=(28087-X_gene[ID].shape[0]*math.floor(28087/X_gene[ID].shape[0]))).astype(int)
            # zeros_chem = np.ones(shape=(3072-X_chem[ID].shape[0]*math.floor(3072/X_chem[ID].shape[0])),dtype=int)

            padded_gene = np.tile(self.X_gene[ID], math.floor(28087/self.X_gene[ID].shape[0]))
            # padded_chem = np.tile(self.X_chem[ID], math.floor(3072/self.X_chem[ID].shape[0]))
            X[i,] = np.expand_dims(np.append(padded_gene, self.X_chem[ID]), axis=1)
            # Store class
            # print(y[i])
            y[i] = self.Y[ID]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        # print(self.indexes)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

# train_params = {'dim': 30286,
#           'batch_size': 8,
#           'shuffle': True,
#           'prows':900,
#           'verbose':False,
#           'data_dir':'./data/',
#           'list_IDs':range(16)}

# # Testing Generators
# training_generator = CDRDataGenerator(**train_params)[0]
# print(training_generator)
