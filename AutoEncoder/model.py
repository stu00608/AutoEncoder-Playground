import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Reshape, Dense, LeakyReLU, BatchNormalization, Dropout, Flatten, Lambda, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

class AutoEncoder():
    # Auto Encoder architecture in Keras.

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

        self.use_batch_norm = params["use_batch_norm"]
        self.use_dropout = params["use_dropout"]

        self.n_layers = len(self.encoder_conv_filters)
        self.z_dim = params["z_dim"]

        self._build()

        optimizer = Adam(learning_rate=params["learning_rate"])

        def reconstruction_loss(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_true, y_pred)))
        
        self.model.compile(optimizer=optimizer, loss=reconstruction_loss)
    
    def _build(self):

        self._build_encoder()
        self._build_decoder()

        self.model = Model(self.encoder_input, self.decoder(self.encoder_output))
    
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

        def sampling(args):
            mu, log_var = args
            # Make a normal distribution.
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            # Base on mu, var_log make a bias distribution.
            return mu + K.exp(log_var/2) * epsilon
        
        self.encoder_output = Lambda(sampling, name="encoder_output")([self.mu, self.log_var])

        self.encoder = Model(self.encoder_input, self.encoder_output)
    
    def _build_decoder(self):
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        
        x = Dense(np.prod(self.shape_before))(decoder_input)
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