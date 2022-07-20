from wandb.keras import WandbCallback
import wandb
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Reshape, Dense, LeakyReLU, BatchNormalization, Dropout, Flatten, Lambda, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow_probability as tfp
import time

def sampling(args):
    mu, log_var = args
    # Make a normal distribution.
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    # Base on mu, var_log make a bias distribution.
    return mu + K.exp(log_var/2) * epsilon



def plot_latent_images(decoder, name, dim, n=10, digit_size=28, additional_msg=''):
    """Plots n x n digit images decoded from the latent space."""
    sample_z_list = np.array([np.random.normal(0, 1, dim) for _ in range(n**2)])
    plt.clf()
    for index, z in enumerate(sample_z_list):
        decoded_digit = decoder(np.array([z]))
        plt.subplot(n, n, index+1)
        plt.axis("off")
        plt.imshow(decoded_digit[0], cmap="Greys_r")

    plt.savefig(name)

class VariationalAutoEncoder():
    # Variational Auto Encoder architecture in Keras.

    def __init__(self, input_shape: np.ndarray, params: dict, encoder_params: dict, decoder_params: dict) -> None:
        """Construct the parameters.

        Parameters
        ----------
        input_shape : np.ndarray
            The input of the model, also will be the shape for output.
        params : dict
            The hyperparamters in config.
        encoder_params : dict
            The layer parameters in config.
        decoder_params : dict
            The layer parameters in config.
        """
        self.input_shape = input_shape
        self.encoder_conv_filters = encoder_params["conv_filters"]
        self.encoder_conv_kernels = encoder_params["conv_kernels"]
        self.encoder_conv_strides = encoder_params["conv_strides"]
        self.decoder_conv_t_filters = decoder_params["conv_t_filters"]
        self.decoder_conv_t_kernels = decoder_params["conv_t_kernels"]
        self.decoder_conv_t_strides = decoder_params["conv_t_strides"]

        self.params = params
        self.use_batch_norm = params["use_batch_norm"]
        self.use_dropout = params["use_dropout"]
        self.use_wandb = params["use_wandb"]

        self.n_layers = len(self.encoder_conv_filters)
        self.z_dim = params["z_dim"]
        self.learning_rate = params["learning_rate"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]

        self.learning_processes = []

        self._build()
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()

        self.model = Model(self.encoder_input, self.decoder(self.encoder_output[0]))

        self.bc_loss = BinaryCrossentropy()

    def _build_encoder(self):
        self.encoder_input = Input(shape=self.input_shape, name="encoder_input")
        # self.model_input = encoder_input
        
        x = self.encoder_input
        for i in range(self.n_layers):
            _conv = Conv2D(
                self.encoder_conv_filters[i], 
                self.encoder_conv_kernels[i], 
                self.encoder_conv_strides[i], 
                padding="same", 
                name="conv_"+str(i+1))
            x = _conv(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)
            
            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=self.use_dropout)(x)
            
        # TODO: Not quite understand.
        self.shape_before = K.int_shape(x)[1:]
        # print(f"shape before was {self.shape_before}")

        x = Flatten()(x)

        self.mu = Dense(self.z_dim, name="mu")(x)
        self.log_var = Dense(self.z_dim, name="log_var")(x)

        # print(K.int_shape(self.mu))

        self.z = Lambda(sampling, name="encoder_output")([self.mu, self.log_var])
        self.encoder_output = [self.z, self.mu, self.log_var]

        self.encoder = Model(self.encoder_input, self.encoder_output)
    
    def _build_decoder(self):
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        # print(np.prod(self.shape_before))
        # print(self.shape_before)
        
        x = Dense(int(np.prod(self.shape_before)))(decoder_input)
        x = Reshape(self.shape_before)(x)

        for i in range(self.n_layers):
            _conv_t = Conv2DTranspose(
                self.decoder_conv_t_filters[i],
                self.decoder_conv_t_kernels[i],
                self.decoder_conv_t_strides[i],
                padding="same",
                name="conv_t_"+str(i+1)
            )
            x = _conv_t(x)

            if i<self.n_layers-1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                x = LeakyReLU()(x)

                if self.use_dropout:
                    x = Dropout(rate=self.use_dropout)(x)
        
            else:
                x = Activation('sigmoid')(x)
        
        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)
    
    def r_loss(self, y_true, y_pred):
        """RMSE"""
        return K.sqrt(K.mean(K.square(K.cast(y_true, "float32") - K.cast(y_pred, "float32"))))
        
    def kl_loss(self, mu, log_var):
        return -0.5 * K.mean(1+log_var-K.square(mu)-K.exp(log_var))
        
    def vae_loss(self, y_true, y_pred):
        r_l = self.r_loss(y_true, y_pred)
        kl_l = self.kl_loss(y_true, y_pred)
        return r_l + kl_l

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, mu, log_var = self.encoder(x, training=True)
            reconstruct = self.decoder(z, training=True)

            # r_loss = self.r_loss(x, reconstruct)
            bc_loss = self.bc_loss(x, reconstruct)
            kl_loss = self.kl_loss(mu, log_var)
            loss = bc_loss + kl_loss

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return bc_loss, kl_loss, loss

    def train(self, train_dataset):

        # callbacks = []

        if self.use_wandb:
            wandb.init(config=self.params, project='vae-mnist-experiment')
            # callbacks.append(WandbCallback())

            encoder_model_name = wandb.run.name+"-encoder.png"
            # encoder_model_fig_path = os.path.join(wandb.run.name, encoder_model_name)
            decoder_model_name = wandb.run.name+"-decoder.png"
            # encoder_model_fig_path = os.path.join(wandb.run.name, decoder_model_name)
            os.mkdir(wandb.run.name)
            fig = plot_model(self.encoder, to_file=encoder_model_name+'.png', show_layer_names=True, show_shapes=True)
            fig = plot_model(self.decoder, to_file=decoder_model_name+'.png', show_layer_names=True, show_shapes=True)
            wandb.log({'Encoder': wandb.Image(encoder_model_name+'.png')})
            wandb.log({'Decoder': wandb.Image(decoder_model_name+'.png')})
            os.remove(encoder_model_name+'.png')
            os.remove(decoder_model_name+'.png')

        self.optimizer = Adam(learning_rate=self.learning_rate)
        # train_acc_metric = RootMeanSquaredError()

        for epoch in range(self.epochs):
            print(f"\nstart epoch {epoch}")

            start_time = time.time()
            for step, x_batch_data in enumerate(train_dataset):
                bc_loss, kl_loss, loss = self.train_step(x_batch_data)

                if step % 200 == 0:
                    print(f"Step {step} | bc_loss : {bc_loss} | kl_loss : {kl_loss} | loss : {loss}")
            
            if self.use_wandb:  
                wandb.log({
                    "bc_loss": bc_loss,
                    "kl_loss": kl_loss,
                    "loss": loss})

                img_name = os.path.join(wandb.run.name, wandb.run.name+f"-distribution-{len(os.listdir(wandb.run.name))}.png")
                plot_latent_images(self.decoder, img_name, self.z_dim, n=8)
                wandb.log({"model_architecture": wandb.Image(img_name)})
                os.remove(img_name)

            print(f"\nTime : {time.time()-start_time}\n\n")

        if self.use_wandb: 
            wandb.finish()