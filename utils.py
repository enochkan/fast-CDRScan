from tqdm import tqdm
import numpy as np

# encode genomic data to binary strings of length 25
# {
#     first 17 digits = mutational position
#     next 4 digits = before ATGC one hot 
#     last 4 digits = after ATGC one hot
# }


def to_onehot(case):
    if case == 'A':
        return '1000'
    elif case == 'T':
        return '0100'
    elif case == 'G':
        return '0010'
    elif case == 'C':
        return '0001'

def to_binary(df):
    bin_strings = ['{0:017b}'.format(int(s[2:-3]))+to_onehot(s[-3])+to_onehot(s[-1]) for s in df['Mutation CDS'].tolist()]
    total = []
    # print('Preprocessing CCLP data... ')
    for string in bin_strings:
        total.append([int(char) for char in string])
    return np.array(total)

def iter_encoder(string, maxlen):
    smiles_set = ['P', 'H', ')', '+', '8', 'I', 't', '6', '#', 'N', 'l', '-', '.', ']', '7', '@', '\\', '3', '9', '[', 'F', 'C', 'r', 'S', 'B', '=', '5', 'O', '/', '1', '2', '4', '(']
    strlen = len(string)
    ret = ''
    for char in range(strlen):
        get_index = smiles_set.index(string[char])
        ret += int(get_index)*'0'+'1'+int(len(smiles_set)-(get_index+1))*'0'
    if strlen < maxlen:
        remain = maxlen-len(string)
        ret += remain*len(smiles_set)*'0'
    return ret

def smiles_to_onehot(df, verbose=0):
    total = []
    # get maximum length
    maxl = len(max(df['SMILES.x'].tolist(), key=len))
    if verbose:
        # get unique set of smiles characters
        smiles_set = list({l for word in df['SMILES.x'].tolist() for l in word})
        print('Unique SMILES characters used for onehot encoding: ')
        print(smiles_set)
        print(maxl)
    bin_strings = [iter_encoder(s, maxl) for s in df['SMILES.x'].tolist()]
    # print('Prepocessing SMILES... ')
    for string in bin_strings:
        total.append([int(char) for char in string])
    return np.array(total)
    
