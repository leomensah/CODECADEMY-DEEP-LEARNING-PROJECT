import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

dataset = pd.read_csv('admissions_data.csv')
# print(dataset.head())
# print(dataset.dtypes)

dataset = dataset.drop(['Serial No.'], axis=1)
features = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]
# print(features.dtypes)
# print(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state=42)
# print(features_train.shape)
# print(features_test.shape)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

my_model = Sequential()
my_input = layers.InputLayer(input_shape = features.shape[1])
my_model.add(my_input)
my_model.add(Dense(64, activation='relu'))
my_model.add(Dense(1))
opt = Adam(learning_rate = 0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer = opt)
history = my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)


# print(dataset.head())
# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()
fig.savefig('static/images/my_plots.png')
fig.savefig('static/images/my_plots.png')