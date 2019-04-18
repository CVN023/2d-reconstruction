from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, UpSampling2D, Reshape
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import regularizers



class FBP(object):
    def __init__(self, width, height, proj_shape, nb_proj, channels, sparse_reg_l1, nb_kernels):
        self.WIDTH = int(width)
        self.HEIGHT = int(height)
        self.PROJ_SHAPE = int(proj_shape)
        self.NB_PROJ = int(nb_proj)
        self.CHANNELS = int(channels)
        self.SHAPE = (self.WIDTH, self.HEIGHT, self.CHANNELS)
        self.OPTIMIZER = Adam()
        self.SPARSE_REG_L1 = sparse_reg_l1
        self.NB_KERNELS = nb_kernels
        self.fbp = self.filter_back_proj()
        self.fbp.compile(loss='mean_absolute_error', optimizer=self.OPTIMIZER)
        
        
        
    def filter_back_proj(self):
        model = Sequential()
        model.add(Conv2D(self.NB_KERNELS, (1, self.PROJ_SHAPE), activation='relu', use_bias=True, padding='same',
                         input_shape=(self.PROJ_SHAPE, self.NB_PROJ, self.CHANNELS)))
        model.add(Conv2D(1, (1, 1), activation='linear', use_bias=False, padding='same'))
        model.add(Flatten())
        model.add(Dense(self.WIDTH*self.HEIGHT, use_bias=False, kernel_regularizer=regularizers.l1(self.SPARSE_REG_L1)))
        model.add(Reshape((self.WIDTH, self.HEIGHT, self.CHANNELS)))
        return model

    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch, print_interval, save_weights_int):

        self.fbp_losses = []
        self.val_losses = []
        for cnt in range(epochs):
            # train filter_back_proj
            random_index =  np.random.randint(0, len(X_train) - batch)
            y = y_train[random_index : random_index + int(batch)].reshape(int(batch), self.WIDTH, self.HEIGHT, self.CHANNELS)
            proj = X_train[random_index : random_index + int(batch)].reshape(int(batch),
                                                                             self.PROJ_SHAPE, self.NB_PROJ, self.CHANNELS)
            fbp_loss = self.fbp.train_on_batch(proj, y)
            
            if cnt % print_interval == 0:
                self.fbp_losses.append(fbp_loss)
                val_loss = self.fbp.evaluate(X_val, y_val, batch_size=batch)
                self.val_losses.append(val_loss)
                print ('epoch: %d, [Reconstruction loss: fbp_loss: %f - Val loss: %f]' % (cnt, fbp_loss, val_loss))
            
            if cnt % save_weights_int == 0:
                self.fbp.save_weights('10kernels_relu_sparse_reg_dense_1e-3_batch32_%d.h5' % (cnt))