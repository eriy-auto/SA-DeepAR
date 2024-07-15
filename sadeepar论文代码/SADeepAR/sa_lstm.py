import math
from functools import partial
import logging
from typing import Optional, Union
from attention import Attention
import numpy as np
from numpy.random import normal
import tensorflow as tf
from keras.layers.merge import *
from keras.models import Sequential
from keras.layers.core import *
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Attention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Layer
from attention_deepar.SADeepAR.phy_loss import gaussian_likelihood
from attention_deepar.SADeepAR._init_ import NNModel
from attention_deepar.SADeepAR.sadp_layers import GaussianLayer

# import torch.nn as nn

logger = logging.getLogger(__name__)


def basic_structure_lstm(n_steps=10, dimensions=1, x_data=None, y_data=None, epochs=100, verbose="auto"):
    optimizer = "adam"
    ouput = 1
    step_len = 1
    input_shape = (n_steps, dimensions)
    print('lstm_input_shape:', input_shape)
    inputs = Input(shape=input_shape)
    model = LSTM(units=32, activation='relu', return_sequences=True, dropout=0.1)(inputs)
    model = muti_head_attention(model, step_len, 1)
    # model = LSTM(units=32, activation='relu', dropout=0.5)(model)
    # model = Dense(10)(model)
    model = Dense(int(4 * (1 + math.log(dimensions))), activation="relu")(model)
    model = Model(inputs, ouput)
    model.compile(optimizer=optimizer)
    model.summary()
    patience = 10
    callback = callbacks.EarlyStopping(monitor="loss", patience=patience)
    x_train = x_data
    y_train = y_data
    model.fit(
        x=x_train, y=y_train,
        batch_size=128,
        # steps_per_epoch=self.steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        # callbacks=[callback]
    )
    return model


def muti_head_attention(_input, d=8, n_attention_head=2):
    """
    实现单层多头机制
    @param _input: 输入 (?, n_feats, n_dim)
    @param d: Q,K,V映射后的维度
    @param n_attention_head: multi-head attention的个数
    """
    attention_heads = []

    for i in range(n_attention_head):
        embed_q = layers.Dense(d)(_input)  # 相当于映射到不同的空间,得到不同的Query
        embed_v = layers.Dense(d)(_input)  # 相当于映射到不同的空间,得到不同的Value
        attention_output = layers.Attention()([embed_q, embed_v])
        # 将每一个head的结果暂时存入
        attention_heads.append(attention_output)

    # 多个head则合并，单个head直接返回
    if n_attention_head > 1:
        muti_attention_output = layers.Concatenate(axis=-1)(attention_heads)
    else:
        muti_attention_output = attention_output
    return muti_attention_output


class DeepAR(NNModel):
    def __init__(self, n_steps, dimensions, steps_per_epoch=50, epochs=100, loss=gaussian_likelihood, optimizer="adam",
                 with_custom_nn_structure=None):
        """Init."""
        super().__init__()
        self.n_steps = n_steps
        self.dimensions = dimensions
        self.inputs, self.z_sample = None, None
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.keras_model = None
        if with_custom_nn_structure:
            self.nn_structure = with_custom_nn_structure
        else:
            self.nn_structure = partial(
                DeepAR.basic_structure_se,
                n_steps=self.n_steps,
                dimensions=self.dimensions,
            )
        self._output_layer_name = "main_output"
        self.get_intermediate = None

    def basic_structure_lstm(self, n_steps=10, dimensions=1, x_data=None, y_data=None, epochs=100, verbose= "auto"):
        ouput = 1
        step_len = 1
        input_shape = (n_steps, dimensions)
        print('lstm_input_shape:', input_shape)
        inputs = Input(shape=input_shape)
        model = LSTM(units=32, activation='relu', return_sequences=True, dropout=0.1)(inputs)
        model = muti_head_attention(model, step_len, 1)
        # model = LSTM(units=32, activation='relu', dropout=0.5)(model)
        # model = Dense(10)(model)
        model = Dense(int(4 * (1 + math.log(dimensions))), activation="relu")(model)
        model = Model(inputs, ouput)
        model.compile(optimizer=self.optimizer)
        model.summary()
        patience = 10
        callback = callbacks.EarlyStopping(monitor="loss", patience=patience)
        x_train = x_data
        y_train = y_data
        model.fit(
            x=x_train, y=y_train,
            batch_size=128,
            # steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            # callbacks=[callback]
        )
        return model

    def basic_structure_se(n_steps=10, dimensions=1):
        step_len = 10
        input_shape = (n_steps, dimensions)
        print('lstm_input_shape:', input_shape)
        inputs = Input(shape=input_shape)
        model = LSTM(units=12, activation='softmax', return_sequences=True, dropout=0.1)(inputs)
        model = muti_head_attention(model, step_len, 1)
        # model = LSTM(units=32, activation='relu', dropout=0.5)(model)
        # model = Dense(10)(model)
        # model = Dense(int(4 * (1 + math.log(dimensions))), activation="softmax")(model)  # relu, softmax
        model = Dense(10, activation="relu")(model)
        # model = Dense(1)(model)
        loc, scale = GaussianLayer(1, name="main_output")(model)
        return input_shape, inputs, [loc, scale]

    def basic_structure(self, n_steps=10, dimensions=1):
        """
        This is the method that needs to be patched when changing NN structure
        :return: inputs_shape (tuple), inputs (Tensor), [loc, scale] (a list of theta parameters
        of the target likelihood).

        Please note that I've made up scaling rules of the hidden layer dimensions.
        """
        step_len = 10
        input_shape = (n_steps, dimensions)
        print('lstm_input_shape:', input_shape)
        inputs = Input(shape=input_shape)
        lstm_layer = LSTM(
            16,
            return_sequences=True,
            dropout=0.1
        )(inputs)
        '''
        lstm_layer = LSTM(
            int(4 * (1 + math.pow(math.log(dimensions), 4))),
            return_sequences=True,
            dropout=0.1,
        )(inputs)
        '''
        x = muti_head_attention(lstm_layer, step_len, 1)
        # x = muti_head_attention(x, 32, 3)
        x = Dense(int(4 * (1 + math.log(dimensions))), activation="relu")(x)
        # x = Flatten()(x)
        # x = Dense(5, activation="relu")(x)
        # x = Dropout(0.1)(x)
        # x = Attention(step_len)(x)
        # x = Dense(32, activation="relu")(x)
        # x = Dropout(0.1)(x)
        # x = Dense(int(4 * (1 + math.log(dimensions))), activation="relu")(x)
        '''
        x = Attention_2(step_dim=1)(x)
        # x = Attention(units=30)(x)
        # x.shape = inputs.shape
        # a = Reshape((n_steps, dimensions))(x)
        x = Dense(int(4 * (1 + math.log(dimensions))), activation="relu")(x)
        '''

        loc, scale = GaussianLayer(1, name="main_output")(x)
        return input_shape, inputs, [loc, scale]

    def fit(
            self,
            epochs: Optional[int] = None,
            verbose: Union[str, int] = "auto",
            patience: int = 20,
            x_data=None,
            y_data=None
    ):
        """Fit model.

        This is called from instantiate and fit().

        Args:
            epochs (Optional[int]): number of epochs to train. If nothing
                defined, take self.epochs. Please the early stopping (patience).
            verbose (Union[str, int]): passed to keras.fit(). Can be
                "auto", 0, or 1.
            patience (int): Number of epochs without without improvement to stop.
        """
        if not epochs:
            epochs = self.epochs
        callback = callbacks.EarlyStopping(monitor="loss", patience=patience)
        x_train = x_data
        y_train = y_data
        self.keras_model.fit(
            x=x_train, y=y_train,
            batch_size=128,
            # steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            # callbacks=[callback]
        )
        '''
        self.keras_model.fit(
            ts_generator(self.ts_obj, self.ts_obj.n_steps),
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[callback],
        )
        '''
        if verbose:
            logger.debug("Model was successfully trained")
        self.get_intermediate = K.function(
            inputs=[self.keras_model.input],
            outputs=self.keras_model.get_layer(self._output_layer_name).output,
        )

    def build_model(self):
        input_shape, inputs, theta = self.nn_structure()
        # print('theta', theta)
        model = Model(inputs, theta[0])
        model.compile(loss=self.loss(theta[1]), optimizer=self.optimizer)
        model.summary()
        self.keras_model = model

    def instantiate_and_fit(
            self,
            epochs: Optional[int] = None,
            verbose: Union[str, int] = "auto",
            do_fit: bool = True,
            x_data=None,
            y_data=None
    ):
        """Compile and train model."""
        self.build_model()
        print('model set!')
        self.fit(verbose=verbose, epochs=epochs, x_data=x_data, y_data=y_data)

    @property
    def model(self):
        return self.keras_model

    def predict_theta_from_input(self, input_list):
        """
        This function takes an input of size equal to the n_steps specified in 'Input' when building the
        network
        :param input_list:
        :return: [[]], a list of list. E.g. when using Gaussian layer this returns a list of two list,
        corresponding to [[mu_values], [sigma_values]]
        """
        if not self.get_intermediate:
            raise ValueError("TF model must be trained first!")

        return self.get_intermediate(input_list)

    def get_sample_prediction(self, sample):
        sample = np.array(sample).reshape(
            (1, self.n_steps, self.dimensions)
        )
        output = self.predict_theta_from_input([sample])
        # print(output)
        output = np.array(output)
        # print(output.shape)
        output = output.reshape(output.shape[0], output.shape[2])
        # output = output.mean(axis=0)
        samples = []
        for mu, sigma in zip(output[0], output[1]):
            sample = normal(
                loc=mu, scale=np.sqrt(sigma), size=1
            )  # self.ts_obj.dimensions)
            samples.append(sample)
        # K.clear_session()
        # print(samples)
        return np.array(samples)
