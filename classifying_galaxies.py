import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app


input_data, labels = load_galaxy_data()
X_train,  X_test, Y_train, Y_test = train_test_split(input_data, labels, test_size = 0.20, random_state = 222, stratify=labels)

data_generator = ImageDataGenerator(rescale=(1/255))
training_iterator = data_generator.flow(X_train, Y_train,batch_size=5)
validation_iterator = data_generator.flow(X_test, Y_test, batch_size=5)
# print(input_data.shape)
# print(labels.shape)
model = tf.keras.Sequential()
#Add input layer
model.add(tf.keras.Input(shape=(128,128,3)))

#Add hidden layers
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation='relu'))

#Add output layers
model.add(tf.keras.layers.Dense(4, activation="softmax"))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

steps_per_epoch = (len(X_train)/5)
validation_steps = len(X_test)/5
model.fit(training_iterator, steps_per_epoch=steps_per_epoch, epochs=8, validation_data = validation_iterator, validation_steps=validation_steps)

model.summary()
