import numpy as np
import efficientnet as efn 
from customloader import CustomLoader
from models import BindingModel
from dataloader import CDRDataGenerator
import tensorflow as tf
import keras.backend as K 
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dropout, Dense
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from utils import to_binary, smiles_to_onehot

data_dir = './data'
verbose = 0
prows = .01
bs = 40
shuffle = True
epochs = 30

def root_mean_squared_error(y_true, y_pred):
    # y_pred = K.print_tensor(y_pred,message='y_pred = ')
    # y_true = K.print_tensor(y_true,message='y_true = ')
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def eval_metrics(y_true, y_pred):
    SS_res =  np.sum(np.square(y_true - y_pred)) 
    SS_tot = np.sum(np.square(y_true - np.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + 1e-07) )

def random_split(data_dir, verbose, prows):
    # import cclp data
        cclp_dl = CustomLoader(dataset='cclp',path=data_dir, extension='tsv', verbose=verbose, prows=prows)
        cclp_dl.load_data()
        cclp_df = cclp_dl.data
        cclp_df = cclp_df[['ID_sample','Mutation Description', 'Mutation CDS']]
        cclp_df.rename(columns={"ID_sample": "key"}, inplace=True)

        # import gdsc data
        gdsc_dl = CustomLoader(dataset='gdsc',path=data_dir, extension='csv', verbose=verbose, prows=prows)
        gdsc_dl.load_data()
        gdsc_df = gdsc_dl.data
        gdsc_df = gdsc_df[['drug_name', 'SMILES.x', 'COSMIC_ID', 'LN_IC50']]
        gdsc_df.rename(columns={"COSMIC_ID": "key"}, inplace=True)
    
        joint = cclp_df.set_index('key').join(gdsc_df.set_index('key'))
        joint = joint.dropna()
        indexNames = joint[ joint['Mutation Description'] != ('Substitution - Missense' or 'Substitution - coding silent' or 'Substitution - Nonsense') ].index
        joint = joint.drop(indexNames)
        joint.reset_index(inplace=True) 

        chem_dim = smiles_to_onehot(joint)[0].shape[0]

        # train, test = train_test_split(joint, test_size=0.2)
        train, valid, test = np.split(joint.sample(frac=1), [int(.8*len(joint)), int(.9*len(joint))])

        return train, valid, test, joint, chem_dim

train, valid, test, joint, chem_dim = random_split(data_dir=data_dir, verbose=verbose, prows=prows)

train_IDs = train.index.values.tolist()
valid_IDs = valid.index.values.tolist()
test_IDs = test.index.values.tolist()

# Parameters
train_params = {'dim': 25+chem_dim,
          'batch_size': bs,
          'shuffle': shuffle,
          'prows':prows,
          'verbose':verbose,
          'data_dir':data_dir,
          'list_IDs':train_IDs}

valid_params = {'dim': 25+chem_dim,
          'batch_size': bs,
          'shuffle': shuffle,
          'prows':prows,
          'verbose':verbose,
          'data_dir':data_dir,
          'list_IDs':valid_IDs}

test_params = {'dim': 25+chem_dim,
          'batch_size': 52,
          'shuffle': False,
          'prows':prows,
          'verbose':verbose,
          'data_dir':data_dir,
          'list_IDs':test_IDs}

# Generators
training_generator = CDRDataGenerator(dataset=joint, **train_params)
validation_generator = CDRDataGenerator(dataset=joint, **valid_params)
test_generator = CDRDataGenerator(dataset=joint, **test_params)

# Design model
# def combined_model(concat, effnet):
#     model = Sequential()
#     model.add(concat)
#     model.add(effnet)
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='linear'))
#     model.summary()
#     return model
    

# model = combined_model(BindingModel(chem_dim=chem_dim)._generate_model(), efn.EfficientNetB1(weights=None, pooling='avg', include_top=False))
model = BindingModel(chemdim=chem_dim)()


# Plot model
plot_model(model, to_file='graph.png')
rmsprop = RMSprop(lr=1e-3, clipnorm=1)
model_checkpoint   = ModelCheckpoint('./model_weights.{epoch:03d}-{val_loss:.4f}-{val_r_square:.4f}.h5',save_best_only=True, monitor='val_loss')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=1e-8, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# mcp_save = ModelCheckpoint('best_mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.compile(optimizer=rmsprop, loss=root_mean_squared_error, metrics=[r_square])
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=epochs,  callbacks=[reduce_lr, early_stopping, model_checkpoint])

# Testing 
scores = model.evaluate_generator(generator=test_generator)
print(scores)