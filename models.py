from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout, Reshape, Lambda
from keras.models import Model, Sequential
from keras.backend import expand_dims, stack, identity, int_shape
import efficientnet as efn 
import tensorflow as tf


class BindingModel:

    def __init__(self, chem_dim):
        # self._model = self._generate_model()
        self.cell_line_size = 28075
        self.drug_size = chem_dim
        self.combined_size = self.cell_line_size+self.drug_size

    def _output_cell_line(self, input_cell_line):
        conv_cell_line_1 = Conv1D(filters=5, kernel_size=700, strides=5, activation='tanh')(input_cell_line)
        maxpool_cell_line_1 = MaxPooling1D(pool_size=5)(conv_cell_line_1)
        conv_cell_line_2 = Conv1D(filters=33, kernel_size=5, strides=2, activation='relu')(maxpool_cell_line_1)
        maxpool_cell_line_2 = MaxPooling1D(pool_size=10)(conv_cell_line_2)
        flatten_cell_line = Flatten()(maxpool_cell_line_2)
        dense_cell_line = Dense(100, activation='relu')(flatten_cell_line)
        dropout_cell_line = Dropout(0.1)(dense_cell_line)
        return dense_cell_line

    def _output_drug(self, input_drug):
        conv_drug_1 = Conv1D(filters=50, kernel_size=200, strides=3, activation='tanh')(input_drug)
        maxpool_drug_1 = MaxPooling1D(pool_size=5)(conv_drug_1)
        conv_drug_2 = Conv1D(filters=30, kernel_size=50, strides=5, activation='relu')(maxpool_drug_1)
        maxpool_drug_2 = MaxPooling1D(pool_size=10)(conv_drug_2)
        flatten_drug = Flatten()(maxpool_drug_2)
        dense_drug = Dense(100, activation='relu')(flatten_drug)
        dropout_drug = Dropout(0.1)(dense_drug)
        return dense_drug

    def _concatenate(self, outputs_cell_line, outputs_drug):
        concatenate = Concatenate()([outputs_cell_line, outputs_drug])
        concat_output = Lambda(lambda x: expand_dims(x, axis=2))(concatenate)
        output = Lambda(lambda x: stack([x,x,x], axis=3))(concat_output)
        return output

    def _generate_model(self):
        combined_input = Input(shape=(self.combined_size, 1,), name='combined_input')

        input_cell_line = Lambda(lambda x: x[:, 0:self.cell_line_size, :], name='cell_line_input')(combined_input)
        output_cell_line = self._output_cell_line(input_cell_line)

        input_drug = Lambda(lambda x: x[:, self.cell_line_size:self.combined_size, :], name='drug_input')(combined_input)
        output_drug = self._output_drug(input_drug)

        output = self._concatenate(output_cell_line, output_drug)

        model = Model(inputs=[combined_input], outputs=output)
        return model

    # def __call__(self, *args, **kwargs):
    #     return self._model


    ## log base 2
    ## < -2: sensitive -> 1
    ## otherwise: resistant -> 0