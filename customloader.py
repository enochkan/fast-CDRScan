import pandas as pd
from pandas import DataFrame
import sys
import os
import random
random.seed(0) 
sys.dont_write_bytecode = True



class CustomLoader():
    def __init__(self, path, dataset='train', verbose=0, extension='csv', prows=None):
        self.verbose = verbose
        self.datapath = path+'/'+dataset+'.'+extension
        self.extension = extension
        self.prows = prows

    def load_data(self):
        if self.verbose:
            print('Reading data...')
        if self.extension == 'tsv':
            if self.prows:
                self.data = pd.read_csv(self.datapath, sep='\t', header=0, skiprows=lambda i: i>0 and random.random() > self.prows)
            else:
                self.data = pd.read_csv(self.datapath, sep='\t', header=0)
            if self.verbose:
                print('Read data successfully ...')
        else:
            if self.prows:
                self.data = pd.read_csv(self.datapath, header=0, skiprows=lambda i: i>0 and random.random() > self.prows)
            else:
                self.data = pd.read_csv(self.datapath, header=0)
            if self.verbose:
                print('Read data successfully ...')
        r, c = self.data.shape
        if self.verbose:
            print('dataset contains: '+str(r)+' rows and '+str(c)+' columns')
            print(self.data.head())
            print('dataset summary...')
            print(self.data.info())
