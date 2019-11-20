from customloader import CustomLoader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import to_binary, smiles_to_onehot
import sys
sys.dont_write_bytecode = True
np.set_printoptions(threshold=sys.maxsize)

class CDRDataset(Dataset):
    """Cosmic Cell-line Project Dataset."""

    def __init__(self, data_dir, transform=None, verbose=0, nrows=None):
        """
        Args:
            tsv_file (string): Path to the tsv file
            data_dir (string): Directory with all the data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # import cclp data
        cclp_dl = CustomLoader(dataset='cclp',path=data_dir, extension='tsv', verbose=verbose, nrows=nrows)
        cclp_dl.load_data()
        cclp_df = cclp_dl.data
        cclp_df = cclp_df[['ID_sample','Mutation Description', 'Mutation CDS']]
        cclp_df.rename(columns={"ID_sample": "key"}, inplace=True)
        if verbose:
            print('CCLP Data trimmed...')
            print(cclp_df.head)

        # import gdsc data
        gdsc_dl = CustomLoader(dataset='gdsc',path=data_dir, extension='csv', verbose=verbose, nrows=nrows)
        gdsc_dl.load_data()
        gdsc_df = gdsc_dl.data
        gdsc_df = gdsc_df[['drug_name', 'SMILES.x', 'COSMIC_ID', 'LN_IC50']]
        gdsc_df.rename(columns={"COSMIC_ID": "key"}, inplace=True)
        if verbose:
            print('GDSC Data trimmed...')
            print(gdsc_df.head)
    
        joint = cclp_df.set_index('key').join(gdsc_df.set_index('key'))
        joint = joint.dropna()
        indexNames = joint[ joint['Mutation Description'] != ('Substitution - Missense' or 'Substitution - coding silent') ].index
        joint = joint.drop(indexNames)
        
        # transform from natural log scale to mmol
        joint['LN_IC50'] = joint['LN_IC50'].apply(lambda x: np.exp(x))

        if verbose:
            print('Combined...')
            print(joint.head)

        self.data = joint
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.data.reset_index(inplace=True) 
        print(self.data.head)

        X_gene = to_binary(self.data)
        X_chem = smiles_to_onehot(self.data)
        Y = self.data.ix[:,5]
        print(Y)

        if self.verbose:
            print('Loaded 1D genomic data with shape... ' + str(X_gene.shape))
            print('Loaded 1D chemical data with shape... ' + str(X_chem.shape))
            print('Loaded IC50 data with shape... ' + str(Y.shape))

        X_gene_sample = X_gene[idx]
        X_chem_sample = X_chem[idx]
        Y_sample = Y[idx]

        # if self.transform:
        #     sample = self.transform(sample)

        return {'genes':X_gene_sample, 'chem':X_chem_sample, 'y':Y_sample}

dat = CDRDataset(data_dir='./data', verbose=1)