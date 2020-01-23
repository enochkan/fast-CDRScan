from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout, Reshape, Lambda
from keras.models import Model, Sequential
from keras.backend import expand_dims, stack, identity, int_shape
import efficientnet as efn 
import tensorflow as tf


class BindingModel:
    def __init__(self, chemdim):
        self._model = self._generate_model(chemdim)

    def _output_cell_line(self, input_cell_line):
        conv_cell_line_1 = Conv1D(filters=5, kernel_size=22, strides=1, activation='tanh')(input_cell_line)
        maxpool_cell_line_1 = MaxPooling1D(pool_size=1)(conv_cell_line_1)
        conv_cell_line_2 = Conv1D(filters=33, kernel_size=2, strides=1, activation='relu')(maxpool_cell_line_1)
        maxpool_cell_line_2 = MaxPooling1D(pool_size=2)(conv_cell_line_2)
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
        # concat_output = Lambda(lambda x: expand_dims(x, axis=2))(concatenate)
        # output = Lambda(lambda x: stack([x,x,x], axis=3))(concat_output)
        output = Dense(1, activation='linear')(concatenate)

        return output

    def _generate_model(self, chemdim):
        cell_line_size = 25
        combined_size = cell_line_size + chemdim
        combined_input = Input(shape=(combined_size, 1,), name='combined_input')

        input_cell_line = Lambda(lambda x: x[:, 0:cell_line_size, :], name='cell_line_input')(combined_input)
        output_cell_line = self._output_cell_line(input_cell_line)

        input_drug = Lambda(lambda x: x[:, cell_line_size:combined_size, :], name='drug_input')(combined_input)
        output_drug = self._output_drug(input_drug)

        output = self._concatenate(output_cell_line, output_drug)

        model = Model(inputs=[combined_input], outputs=output)
        return model

    def __call__(self, *args, **kwargs):
        return self._model