# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (December-2023)

## THIS CODE EXECUTES THE BAYESIAN OPTIMIZATION (BO) ALGORITHM FOR FINDING THE PRE-RELAXED LATTICE PARAMETERS.
# PHASE 3: BAYESIAN OPTIMIZATION (BO) MODEL

from keras.layers import Input, Dense, Flatten, Conv2D, Droupout, MaxPooling2D
from keras.models import Model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Build Regressor Model for Predicting Total DFT Computed Energy (per atom)
energy_input = Input(shape=(X_samples.shape[1], X_samples.shape[2], X_samples.shape[3]))
x = Conv2D(8, (3, 3), activation='relu', strides= 1, padding='same')(energy_input)
x = Conv2D(16, (3, 3), activation='relu', strides=1, padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(4, activation='relu')(x)
energy_target = Dense(1, activation='linear')(x)

opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, decay=1e-4/200)
energy_model = Model(energy_input, energy_target)
energy_model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mae'])

# Train Model
X_train, X_val, y_train, y_val = train_test_split(X_samples, energy_per_atom, test_size=0.2, random_state=72)
energy_history = energy_model.fit(X_train, y_train, epochs=350, verbose=0, validation_data=(X_val, y_val))

# Save Trained Model
energy_model.save('energy_model.h5')

# Install Scikit Optimize Library for Executing Bayesian Optimization
# !pip install scikit-optimize

# Import Bayesian Optimization Library
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize

samp = i # 'X_novel' is the newly generated/decoded material from Phase 2 (i.e. A-GAN modeling phase) in mesh-grid form

# Note that 'tolerances' are used to further guide the search operation.
# Tolerances are defined using chemical knowledge from intuition derived in the dataset distribution.

# For CUBIC CRYSTAL SYSTEM (i.e. One search-space dimensionality on the lattice edge)
x0 = Real(name='x0_', low=(low_tolerance), high=(high_tolerance))
dimensions = [x0]

@use_named_args(dimensions=dimensions)
def my_objective_function(x0_):
  bo_latt = np.array((x0_,x0_,x0_)).reshape(1,3)
  bo_latt = scaler_LV.transform(bo_latt).reshape(1,3).astype('float32')
  zeropad = np.zeros((1,1))
  bo_latt = np.concatenate((bo_latt,zeropad),1)
  bo_label = np.concatenate(((X_novel[samp,:,:,0].reshape(256,4)[:101,]),bo_latt,
                          (X_novel[samp,:,:,0].reshape(256,4)[102:,])),0).reshape(32,32,1)
  bo_image = np.concatenate((bo_label, X_novel[samp,:,:,1:]),2)
  bo_image = bo_image.reshape(1,32,32,3)

  return float(energy_model.predict(bo_image))

# For MONOCLINIC CRYSTAL SYSTEM (i.e. Four search-space dimensionality on the lattice edge)
x0 = Real(name='x0_', low=(low_tolerance0), high=(high_tolerance0))
x1 = Real(name='x1_', low=(low_tolerance1), high=(high_tolerance1))
x2 = Real(name='x2_', low=(low_tolerance2), high=(high_tolerance2))
x3 = Real(name='x3_', low=(low_tolerance3), high=(high_tolerance3))
dimensions = [x0,x1,x2,x3]

@use_named_args(dimensions=dimensions)
def my_objective_function(x0_,x1_,x2_,x3_):
  bo_latt = np.array((x0_,x1_,x2_)).reshape(1,3)
  bo_latt = scaler_LV.transform(bo_latt).reshape(1,3).astype('float32')
  zeropad = np.zeros((1,1))
  bo_latt = np.concatenate((bo_latt,zeropad),1)
  bo_ang = np.array((90,x3_,90)).reshape(1,3)
  bo_ang = scaler_ANG.transform(bo_ang).reshape(1,3).astype('float32')
  zeropad = np.zeros((1,1))
  bo_ang = np.concatenate((bo_ang,zeropad),1)
  bo_label = np.concatenate(((X_novel[samp,:,:,0].reshape(256,4)[:101,]),bo_latt,
                             bo_ang,(X_novel[samp,:,:,0].reshape(256,4)[103:,])),0).reshape(32,32,1)
  bo_image = np.concatenate((bo_label, X_novel[samp,:,:,1:]),2)
  bo_image = bo_image.reshape(1,32,32,3)

  return float(energy_model.predict(bo_image))

# For ORTHORHOMBIC CRYSTAL SYSTEM (i.e. Three search-space dimensionality on the lattice edge)
x0 = Real(name='x0_', low=(low_tolerance0), high=(high_tolerance0))
x1 = Real(name='x1_', low=(low_tolerance1), high=(high_tolerance1))
x2 = Real(name='x2_', low=(low_tolerance2), high=(high_tolerance2))
dimensions = [x0,x1,x2]

@use_named_args(dimensions=dimensions)
def my_objective_function(x0_,x1_,x2_):
  bo_latt = np.array((x0_,x1_,x2_)).reshape(1,3)
  bo_latt = scaler_LV.transform(bo_latt).reshape(1,3).astype('float32')
  zeropad = np.zeros((1,1))
  bo_latt = np.concatenate((bo_latt,zeropad),1)
  bo_label = np.concatenate(((X_novel[samp,:,:,0].reshape(256,4)[:101,]),bo_latt,
                          (X_novel[samp,:,:,0].reshape(256,4)[102:,])),0).reshape(32,32,1)
  bo_image = np.concatenate((bo_label, X_novel[samp,:,:,1:]),2)
  bo_image = bo_image.reshape(1,32,32,3)

  return float(energy_model.predict(bo_image))

# For TETRAGONAL AND TRIGONAL CRYSTAL SYSTEM (i.e. Two search-space dimensionality on the lattice edge)
x0 = Real(name='x0_', low=(low_tolerance0), high=(high_tolerance0))
x1 = Real(name='x1_', low=(low_tolerance1), high=(high_tolerance1))
dimensions = [x0,x1]

@use_named_args(dimensions=dimensions)
def my_objective_function(x0_,x1_):
  bo_latt = np.array((x0_,x0_,x1_)).reshape(1,3)
  bo_latt = scaler_LV.transform(bo_latt).reshape(1,3).astype('float32')
  zeropad = np.zeros((1,1))
  bo_latt = np.concatenate((bo_latt,zeropad),1)
  bo_label = np.concatenate(((X_novel[samp,:,:,0].reshape(256,4)[:101,]),bo_latt,
                          (X_novel[samp,:,:,0].reshape(256,4)[102:,])),0).reshape(32,32,1)
  bo_image = np.concatenate((bo_label, X_novel[samp,:,:,1:]),2)
  bo_image = bo_image.reshape(1,32,32,3)

  return float(energy_model.predict(bo_image))

# Execute Gaussian Process (GP) Minimization
result = gp_minimize(my_objective_function, dimensions, n_calls=200, n_initial_points=40,
                     acq_func='EI', random_state=7, n_jobs=-1)
print(result.fun, result.x)  
