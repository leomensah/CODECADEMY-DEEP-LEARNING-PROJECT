import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

data_generator = ImageDataGenerator(rescale=(1/255))

training_data = data_generator.flow_from_directory('augmented-data/train', class_mode='categorical', color_mode='grayscale', batch_size=16)

validation_data = data_generator.flow_from_directory('augmented-data/test', class_mode='categorical', color_mode='grayscale', batch_size=16)


model = Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))
model.add(tf.keras.layers.Conv2D(2, 3, strides=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5,5)))
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dense(8, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

steps_per_epoch = (training_data.samples)/16
validataion_steps = (validation_data.samples)/16
history = model.fit(training_data, steps_per_epoch=steps_per_epoch, epochs=5, validation_data = validation_data, validation_steps = validataion_steps)


# Do Matplotlib extension below
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

#use this to keep plot from overlapping
fig.tight_layout()
fig.show()

# use this savefig call at the end of your graph instead of using plt.show()
plt.savefig('static/images/my_plots.png')

