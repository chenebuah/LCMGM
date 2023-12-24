# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (December-2023)

## THIS CODE EXECUTES THE AUXILIARY GENERATIVE ADVERSARIAL NETWORK FOR GENERATING SYNTHETIC LATENT SPACE VECTORS THAT CONFORM TO A LATTICE CONFIGURATION.
# PHASE 2: AUXILIARY GENERATIVE ADVERSARIAL NETWORK (A-GAN) MODEL

# Preprocess Encoded Latent Vectors and Lattice Constraints
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Define and isolate 'z_space', which is the interested/sampled region from a 2D (dimen_1 versus dimen_2) plot in the latent space, 'z'.

# xlim = [x_min, x_max]
# ylim = [y_min, y_max]
# dimen_1 = x_axis
# dimen_2 = y_axis

test_space = (z[:,dimen_1]>(np.array(xlim))[0,]) & (z[:,dimen_1]<(np.array(xlim))[1,]) & (z[:,dimen_2]>(np.array(ylim))[0,]) & (z[:,dimen_2]<(np.array(ylim))[1,])
z_space = np.array([z[i] for i in (np.where(test_space)[0])])

# Scale all encoded points within -1 and 1
scaler_z_gan = MinMaxScaler(feature_range=(-1, 1)) 
scaler_z_gan.fit(z_space)
x_train_=(scaler_z_gan.transform(z_space))

# Append Lattice Constraints in Atomic Coordinates, Edge Vectors, and Angles.
AC = np.asarray(pd.read_csv('AC_full.csv').astype('float32'))[:,1:]
AC = AC.reshape(AC.shape[0],80,3)
AC = (np.delete(AC, (rmv), axis=0))
AC = AC[:,:40,]

params = np.asarray(pd.read_csv('params_full.csv').astype('float32'))

LV = params[:,1:4]
scaler_LV = MinMaxScaler()
scaler_LV.fit(LV)
LV=(scaler_LV.transform(LV)).reshape(PROP_array.shape[0],1,3)
LV = (np.delete(LV, (rmv), axis=0))

ANG = params[:,4:]
scaler_ANG = MinMaxScaler()
scaler_ANG.fit(ANG)
ANG=(scaler_ANG.transform(ANG)).reshape(PROP_array.shape[0],1,3)
ANG = (np.delete(ANG, (rmv), axis=0))

y_train_ = np.concatenate((LV, ANG, AC), 1)
y_train_ = np.array([y_train_[i] for i in (np.where(test_space)[0])]) # 'test_space' are the indices that correspond to 'z_space'

x_train, trainy_ = shuffle(x_train_, y_train_, random_state=7)

from keras.initializers import RandomNormal
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.layers import Input, Dense, LeakyReLU, Activation, Flatten, Reshape, BatchNormalization, Conv2D, Conv1D, Embedding, Concatenate

# Define the Discriminator/Critic Model
def define_discriminator(in_shape=(256,1)):

  in_image = Input(shape=(256,))
  root = Dense(512, activation=LeakyReLU(alpha=0.2))(in_image)
  root = BatchNormalization()(root)
  disc = Reshape((16,16,2))(root)
  fe = Conv2D(4, (3,3), padding='same', activation=LeakyReLU(alpha=0.2))(disc)
  fe = BatchNormalization()(fe)
  fe = Conv2D(16, (3,3), padding='same', activation=LeakyReLU(alpha=0.2))(fe)
  fe = BatchNormalization()(fe)
  fe = Flatten()(fe)
  fe = Dense(64, activation=LeakyReLU(alpha=0.2))(fe)
  fe = BatchNormalization()(fe)
  fe = Dense(16, activation=LeakyReLU(alpha=0.2))(fe)
  fe = BatchNormalization()(fe)

  # Real/Fake Output
  out1 = Dense(1, activation='sigmoid')(fe)

  # Auxiliary Lattice Regressor Output
  pred = Dense(1134, activation=LeakyReLU(alpha=0.2))(root)
  fe2 = BatchNormalization()(pred)
  fe2 = Reshape((42,27))(fe2)
  fe2 = Conv1D(9, (3,), padding='same', activation=LeakyReLU(alpha=0.2))(fe2)
  fe2 = BatchNormalization()(fe2)
  out2 = Conv1D(3, (3,), padding='same', activation='sigmoid')(fe2)

  # Define and Compile Model
  model = Model(in_image, [out1, out2])
  opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, decay=1e-4/200)
  model.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer=opt)
  return model

model = define_discriminator()


# Define the Stand-Alone Generator Model
def define_generator(noise_dim):

  in_label = Input(shape=(42,3)) # Lattice Constraint
  li = Conv1D(24, (3,), padding='same', activation=LeakyReLU(alpha=0.2))(in_label)
  li = BatchNormalization()(li)
  li = Flatten()(li)
  li = Dense(256, activation=LeakyReLU(alpha=0.2))(li)
  li = Reshape((256, 1))(li)
  in_lat = Input(shape=(100,)) # Random noise vector
  gen = Dense(256, activation=LeakyReLU(alpha=0.2))(in_lat)
  gen = Reshape((256,1))(gen)
  merge = Concatenate()([gen, li])
  gen = Reshape((16,16,2))(merge)
  gen = Conv2D(4, (3,3), padding='same', activation=LeakyReLU(alpha=0.2))(gen)
  gen = BatchNormalization()(gen)
  gen = Conv2D(16, (3,3), padding='same', activation=LeakyReLU(alpha=0.2))(gen)
  gen = BatchNormalization()(gen)
  gen = Flatten()(gen)
  gen = Dense(512, activation=LeakyReLU(alpha=0.2))(gen)
  gen = BatchNormalization()(gen)

  # Synthetized Vector Output
  out_layer = Dense(256, activation='tanh')(gen)

  # Define Model
  model = Model([in_lat, in_label], out_layer)
  return model

noise_dim = 100
model = define_generator(noise_dim)

# Define and Compile Adversarial Network with Passive Discriminator Weights
def define_gan(g_model, d_model):

  for layer in d_model.layers:
    if not isinstance(layer, BatchNormalization):
      layer.trainable = False

  gan_output = d_model(g_model.output)
  model = Model(g_model.input, gan_output)
  opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, decay=1e-4/200)
  model.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer=opt)
  return model


# Load Real Vectors/Samples
def load_real_samples():
  X = x_train
  trainy = trainy_
  return [X, trainy]

# Select Real Samples from 'z_space'
def generate_real_samples(dataset, n_samples):
	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = ones((n_samples, 1))
	return [X, labels], y

# Generate Random Noise as Input into the Generator
def generate_noise_points(noise_dim, n_samples):
	x_input = randn(noise_dim * n_samples)
	z_input = x_input.reshape(n_samples, noise_dim)
	idx_ = randint(0, trainy_.shape[0], n_samples)
	labels = trainy_[idx_,]
	return [z_input, labels]

# Use Generator to Produce Fake Vectors With Class Labels
def generate_fake_samples(generator, noise_dim, n_samples):
	z_input, labels_input = generate_noise_points(noise_dim, n_samples)
	images = generator.predict([z_input, labels_input])
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# Train Generator and Discriminator
def train(g_model, d_model, gan_model, dataset, noise_dim, n_epochs=100, n_batch=64):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	n_steps = bat_per_epo * n_epochs
	half_batch = int(n_batch / 2)

	for i in range(n_steps):
		[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
		_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
		[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, noise_dim, half_batch)
		_,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
		[z_input, z_labels] = generate_noise_points(noise_dim, n_batch)
		y_gan_ = ones((n_batch, 1))
		_,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan_, z_labels])
		print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))

	g_model.save('AGAN_model.h5')
 
noise_dim = 100
discriminator = define_discriminator()
generator = define_generator(noise_dim)
gan_model = define_gan(generator, discriminator)
dataset = load_real_samples()

# Train Model
train(generator, discriminator, gan_model, dataset, noise_dim)

# Generate New Naterials
from numpy import asarray
from keras.models import load_model

model = load_model('AGAN_model.h5')
n_examples = 500

# generate random noise and labels.
noise_points, labels = generate_noise_points(noise_dim, n_examples)

# Note that the labels are the learnt lattice configuration, and can be selected by a user to confrom to a specific material prototype

# generate images
X  = model.predict([noise_points, labels])
novel_material = scaler_z_gan.inverse_transform(X)
