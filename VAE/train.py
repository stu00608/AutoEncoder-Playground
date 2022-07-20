import os
import yaml
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset
from model import VariationalAutoEncoder

if __name__ == "__main__":

    if not os.path.exists("result"):
        os.mkdir("result")

    config = yaml.safe_load(open('config.yaml', 'r'))
    params = config['params']
    encoder_params = config['encoder']
    decoder_params = config['decoder']

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train.astype(np.float64) / 255.
    x_train = np.expand_dims(x_train, axis=3)
    x_test.astype(np.float64) / 255.
    x_test = np.expand_dims(x_test, axis=3)

    train_dataset = Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(1024).batch(params["batch_size"])

    model = VariationalAutoEncoder(
        input_shape=x_train[0].shape, 
        params=params, 
        encoder_params=encoder_params, 
        decoder_params=decoder_params)

    model.train(train_dataset)