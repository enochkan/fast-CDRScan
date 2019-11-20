import pandas as pd
from pandas import DataFrame
import sys
import os
sys.dont_write_bytecode = True



class CustomLoader():
    def __init__(self, path, dataset='train', verbose=0, extension='csv', nrows=None):
        self.verbose = verbose
        self.datapath = path+'/'+dataset+'.'+extension
        self.extension = extension
        self.nrows = nrows

    def load_data(self):
        if self.verbose:
            print('Reading data...')
        if self.extension == 'tsv':
            if self.nrows:
                self.data = pd.read_csv(self.datapath, sep='\t', header=0, nrows=self.nrows)
            else:
                self.data = pd.read_csv(self.datapath, sep='\t', header=0)
            if self.verbose:
                print('Read data successfully ...')
        elif self.extension == 'xlsx':
            if self.nrows:
                self.data = pd.read_excel(self.datapath, header=0, nrows=self.nrows)
            else:
                self.data = pd.read_excel(self.datapath, header=0)
            if self.verbose:
                print('Read data successfully ...')
        else:
            if self.nrows:
                self.data = pd.read_csv(self.datapath, header=0, nrows=self.nrows)
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
