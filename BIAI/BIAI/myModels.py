import tensorflow as tf
from tensorflow import keras

def make1VGGModel():
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	
	optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def make2VGGModel():
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	
	optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def make3VGGModel():
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	
	optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
